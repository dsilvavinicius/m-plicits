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
import shutil
import time
import yaml
import numpy as np
import open3d as o3d
import torch
from torch.nn.utils import parameters_to_vector
from i3d.dataset import _read_ply
from i3d.loss_functions import loss_add_detail_fine
from i3d.model import SIREN
from i3d.util import from_pth
from utils import calc_deltas, create_training_data_on_neighborhood


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

    shutil.copy(args.configpath, osp.join(args.outputpath, "config.yaml"))

    trainingcfg = config["training"]
    SEED = 668123
    EPOCHS = trainingcfg["epochs"] if not args.nepochs else args.nepochs
    BATCH = trainingcfg["batchsize"]
    COARSE_MODEL_PATH = trainingcfg["coarse_model"]
    RES_MODEL_PATH = trainingcfg["level1"]

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    devstr = args.device
    if "cuda" in devstr and not torch.cuda.is_available():
        devstr = "cpu"
        print("No CUDA available devices found on system. Using CPU.")

    device = torch.device(devstr)
    samplingcfg = config.get("sampling", None)

    mesh, vertices = _read_ply(args.meshpath, with_curvatures=False)
    vertices = vertices.to(device)

    N = vertices.shape[0]
    nsteps = round(EPOCHS * (N / BATCH))
    warmup_steps = nsteps // 10
    print(f"Total # of training steps = {nsteps}")

    # Create a raycasting scene to perform the SDF querying
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    # coarse model
    coarse_model = from_pth(COARSE_MODEL_PATH).eval().to(device)
    print("==================== Coarse Model ====================")
    print(coarse_model)

    # res model
    res_model = from_pth(RES_MODEL_PATH).eval().to(device)
    print("==================== Level 1 ====================")
    print(res_model)

    # TODO RECONSTRUCT MEDIUM
    # create_mesh(res_model, "medium.ply", device=device, N=400)

    # approximating the max_dist from the coarse level set to the GT point cloud
    with torch.no_grad():
        dist2coarseSDF = coarse_model(vertices[..., :3])["model_out"]
        resSDF = res_model(vertices[..., :3])["model_out"]

    max_dist = torch.max(torch.abs(dist2coarseSDF + resSDF))
    max_delta = max_dist

    use_neigh_delta = trainingcfg["neighborhood_delta"]
    dithering_sampling = trainingcfg["dithering_sampling"]
    # Create the model and optimizer
    netcfg = config["network"]
    model = SIREN(
        netcfg["in_coords"],
        netcfg["out_coords"],
        hidden_layer_config=netcfg["hidden_layers"],
        w0=netcfg["omega_0"] if not args.omega0 else args.omega0,
        ww=netcfg["omega_w"] if not args.omegaW else args.omegaW
    ).to(device)
    print("==================== Final Model ====================")
    print(model)
    print("# parameters =", parameters_to_vector(model.parameters()).numel())

    nonsurf = round(BATCH * 0.5)
    noffsurf = round(BATCH * 0.5)

    loss_fn = loss_add_detail_fine(coarse_model, res_model)

    DELTA_FRACS = [1.01, 1.05, 1.1, 1.2, 1.3, 1.5, 2, 5]
    deltas = [max_delta * i for i in DELTA_FRACS]
    for i, delta in enumerate(deltas):
        print(f"========== {DELTA_FRACS[i]} * max_delta = {delta} ==========")
        optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

        # computing an tubular neighborhood (the deltas) of the point cloud
        vertices_copy = calc_deltas(
            vertices, scene, max_delta=delta, n_iters=100,
            device=device
        )

        training_loss = {}
        best_loss = torch.inf
        best_weights = None

        start_training_time = time.time()
        for step in range(nsteps):
            samples = create_training_data_on_neighborhood(
                vertices_copy,
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
        print(f"training took {training_time} s")
        model.update_omegas(w0=1, ww=None)
        torch.save(
            model.state_dict(),
            osp.join(args.outputpath, f"weights_{DELTA_FRACS[i]}.pth")
        )
        model.w0 = netcfg["omega_0"] if not args.omega0 else args.omega0
        model.ww = netcfg["omega_w"] if not args.omegaW else args.omegaW
        model.load_state_dict(best_weights)
        model.update_omegas(w0=1, ww=None)
        torch.save(
            model.state_dict(),
            osp.join(args.outputpath, f"best_{DELTA_FRACS[i]}.pth")
        )

        # Reseting for next iteration.
        print(netcfg["omega_0"], netcfg["omega_w"])
        model.update_omegas(
            w0=netcfg["omega_0"] if not args.omega0 else args.omega0,
            ww=netcfg["omega_w"] if not args.omegaW else args.omegaW
        )
        model.reset_weights()
        model.train()
        print(model)
