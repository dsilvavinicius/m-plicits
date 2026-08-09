#!/usr/bin/env python
# coding: utf-8

"""
Experiments with calculating the SDF for a batch of points and reusing it for N
iterations. Most of its behaviour was replicated in the i3d package already,
but we kept this here for historical reasons and to facilitate experiments.
With the exception of the loss function and model definition, this script
should be (and remain) independent of the main i3d package.
"""

import argparse
import copy
import os
import os.path as osp
import time
import yaml
import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
from torch.nn.utils import parameters_to_vector
from i3d.dataset import _read_ply
from i3d.loss_functions import loss_sdf_on_neighborhood, loss_add_detail_fine
from i3d.model import SIREN
from i3d.util import from_pth
from utils import create_training_data_on_neighborhood, calc_deltas
import gc
import csv  # Import csv module for writing the CSV file


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description="Experiments with SDF querying at regular intervals."
    )
    parser.add_argument(
        "meshpath",
        help="Path to the mesh to use for training. We only handle PLY files."
    )
    parser.add_argument(
        "outputpath",
        help="Path to the output folder (will be created if necessary)."
    )
    parser.add_argument(
        "configpath",
        help="Path to the configuration file with the network's description."
    )
    parser.add_argument("coarsepath", help="Path to the coarse PTH file.")
    parser.add_argument("mediumpath", help="Path to the medium PTH file.")
    parser.add_argument(
        "--device", "-d", type=str, default="cuda:0",
        help="The device to perform the training on. Uses CUDA:0 by default."
    )
    parser.add_argument(
        "--nepochs", "-n", type=int, default=0,
        help="Number of training epochs for each mesh."
    )
    parser.add_argument(
        "--omega0", "-o", type=int, default=0,
        help="SIREN Omega 0 parameter."
    )
    parser.add_argument(
        "--omegaW", "-w", type=int, default=0,
        help="SIREN Omega 0 parameter for hidden layers."
    )
    parser.add_argument(
        "--batchsize", "-b", type=int, default=0,
        help="# of points to fetch per iteration. By default, uses the # of mesh vertices."
    )
    args = parser.parse_args()

    if not osp.exists(args.configpath):
        raise FileNotFoundError(
            f"Experiment configuration file \"{args.configpath}\" not found."
        )

    if not osp.exists(args.meshpath):
        raise FileNotFoundError(
            f"Mesh file \"{args.meshpath}\" not found."
        )

    with open(args.configpath, "r") as fin:
        config = yaml.safe_load(fin)

    print(f"Saving results in {args.outputpath}")
    if not osp.exists(args.outputpath):
        os.makedirs(args.outputpath)

    trainingcfg = config["training"]
    SEED = 668123
    EPOCHS = trainingcfg.get("epochs", 100)
    if args.nepochs:
        EPOCHS = args.nepochs
        config["training"]["epochs"] = args.nepochs

    BATCH = trainingcfg.get("batchsize", 0)
    if args.batchsize:
        BATCH = args.batchsize
        config["training"]["batchsize"] = args.batchsize

    COARSE_MODEL_PATH = args.coarsepath  # trainingcfg["coarse_model"]
    MEDIUM_MODEL_PATH = args.mediumpath  # trainingcfg["level1"]

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    devstr = args.device
    if "cuda" in devstr and not torch.cuda.is_available():
        devstr = "cpu"
        print("No CUDA available devices found on system. Using CPU.")

    device = torch.device(devstr)

    samplingcfg = config.get("sampling", None)
    try:
        mesh, vertices = _read_ply(args.meshpath, with_curvatures=False)
    except:
        mesh = o3d.io.read_triangle_mesh(args.meshpath).remove_duplicated_vertices().remove_degenerate_triangles().remove_non_manifold_edges()
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)

        mesh = o3d.t.geometry.TriangleMesh(o3c.Device("CPU:0"))
        mesh.vertex["positions"] = o3c.Tensor(vertices[:, :3], dtype=o3c.float32)
        mesh.vertex["normals"] = o3c.Tensor(vertices[:, 3:6], dtype=o3c.float32)
        mesh.triangle["indices"] = o3c.Tensor(faces, dtype=o3c.int32)

        vertices = torch.from_numpy(np.hstack((vertices, normals))).requires_grad_(False)

    vertices = vertices.to(device)

    if not BATCH or BATCH > vertices.shape[0]:
        BATCH = 2 * vertices.shape[0]

    N = vertices.shape[0]
    nsteps = round(EPOCHS * (N / BATCH))
    warmup_steps = nsteps // 10
    print(f"Total # of training steps = {nsteps}")

    # Create a raycasting scene to perform the SDF querying
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    # Load coarse model
    coarse_model = from_pth(COARSE_MODEL_PATH).eval().to(device)
    print("==================== Coarse Model ====================")
    print(coarse_model)

    # Load level 1 (residual) model
    res_model = from_pth(MEDIUM_MODEL_PATH).eval().to(device)
    print("==================== Level 1 ====================")
    print(res_model)

    # Approximating the max distance from the coarse level set to the GT point cloud
    with torch.no_grad():
        dist2coarseSDF = coarse_model(vertices[..., :3])["model_out"]
        resSDF = res_model(vertices[..., :3])["model_out"]

    max_dist = torch.max(torch.abs(dist2coarseSDF + resSDF))
    max_delta = max_dist
    delta = max_dist * 1.3

    # Delete the tensors explicitly
    del dist2coarseSDF
    del resSDF

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_neigh_delta = trainingcfg["neighborhood_delta"]
    dithering_sampling = trainingcfg["dithering_sampling"]

    # Compute the tubular neighborhood (the deltas) of the point cloud
    if use_neigh_delta:
        vertices = calc_deltas(
            vertices, scene, max_delta=delta, n_iters=100, device=device
        )

    # Create the model and the optimizer
    netcfg = config["network"]
    model = SIREN(
        netcfg["in_coords"],
        netcfg["out_coords"],
        hidden_layer_config=netcfg["hidden_layers"],
        w0=netcfg["omega_0"] if not args.omega0 else args.omega0,
        ww=netcfg["omega_w"] if not args.omegaW else args.omegaW
    ).to(device)
    print(model)

    num_parameters = parameters_to_vector(model.parameters()).numel()
    print("# parameters =", parameters_to_vector(model.parameters()).numel())
    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
    training_loss = {}

    nonsurf = round(BATCH * 0.5)
    best_loss = torch.inf
    best_weights = None

    if netcfg["residual"]:
        print("========== Residual training ==========")
        if "constraint_weights" not in config["training"]:
            constraint_weights = loss_add_detail_fine.DEFAULT_WEIGHTS
            config["training"]["constraint_weights"] = loss_add_detail_fine.DEFAULT_WEIGHTS
        else:
            constraint_weights = config["training"]["constraint_weights"]
        loss_fn = loss_add_detail_fine(coarse_model, res_model, constraint_weights)
    else:
        print("========== Non-residual training ==========")
        loss_fn = loss_sdf_on_neighborhood

    with open(osp.join(args.outputpath, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    start_training_time = time.time()
    for step in range(nsteps):
        samples = create_training_data_on_neighborhood(
            vertices,
            n_on_surf=nonsurf,
            device=device,
            delta=delta,
            use_neigh_delta=use_neigh_delta,
            dithering_sampling=dithering_sampling
        )

        trainingpts = samples["on_surf"][0]
        trainingnormals = samples["on_surf"][1]
        trainingsdf = samples["on_surf"][2]

        gt = {
            "sdf": trainingsdf.float().unsqueeze(1),
            "normals": trainingnormals.float(),
        }

        optim.zero_grad(set_to_none=True)
        y = model(trainingpts)
        loss = loss_fn(y, gt)

        running_loss = torch.zeros((1, 1), device=device)
        for k, v in loss.items():
            running_loss += v
            if k not in training_loss:
                training_loss[k] = [v.detach()]
            else:
                training_loss[k].append(v.detach())

        running_loss.backward()
        optim.step()

        if step > warmup_steps and running_loss.item() < best_loss:
            best_weights = copy.deepcopy(model.state_dict())
            best_loss = running_loss.item()

        if not step % 100 and step > 0:
            print(f"Step {step} --- Loss {running_loss.item()}")

    training_time = time.time() - start_training_time
    print(f"Training took {training_time} s")
    print(f"TRAINING_TIME_S:{training_time}")

    # Save several versions of the model
    torch.save(model.state_dict(), osp.join(args.outputpath, "weights_with_w0.pth"))
    model.update_omegas(w0=1, ww=None)
    torch.save(model.state_dict(), osp.join(args.outputpath, "weights.pth"))
    torch.save(best_weights, osp.join(args.outputpath, "best_with_w0.pth"))

    model.w0 = netcfg["omega_0"] if not args.omega0 else args.omega0
    model.ww = netcfg["omega_w"] if not args.omegaW else args.omegaW
    model.load_state_dict(best_weights)
    model.update_omegas(w0=1, ww=None)
    best_model_path = osp.join(args.outputpath, "best.pth")
    torch.save(model.state_dict(), best_model_path)

    # --- Statistics Calculation and CSV Writing ---

    # Calculate the file size of the best model in bytes and convert to kilobytes (KB)
    file_size_bytes = osp.getsize(best_model_path)
    file_size_kb = file_size_bytes / 1024  # Convert bytes to KB
    file_size_kb = round(file_size_kb, 2)   # Optionally, round for neatness

    # Write the collected statistics into a CSV file in the output folder
    metrics_csv = osp.join(args.outputpath, "metrics.csv")
    with open(metrics_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write CSV header
        csv_writer.writerow(["Training Time (s)", "Num Parameters", "Model File Size (KB)"])
        # Write the actual values
        csv_writer.writerow([training_time, num_parameters, file_size_kb])

    print(f"Metrics saved to {metrics_csv}")
