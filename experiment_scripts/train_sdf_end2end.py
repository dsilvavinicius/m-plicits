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
from i3d.dataset import PointCloud
from i3d.loss_functions import loss_add_detail_sitzmann
from i3d.model import SIREN
from i3d.util import from_pth
import gc
import csv


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description="Experiments with end-to-end residual SDF training."
    )
    parser.add_argument(
        "meshpath",
        help="Path to the mesh to use for training. We only handle PLY files.",
    )
    parser.add_argument(
        "outputpath", help="Path to the output folder (will be created if necessary)."
    )
    parser.add_argument(
        "configpath",
        help="Path to the configuration file with the network's description.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        help="The device to perform the training on. Uses CUDA:0 by default.",
    )
    parser.add_argument(
        "--nepochs",
        "-n",
        type=int,
        default=0,
        help="Number of training epochs for each mesh.",
    )
    parser.add_argument(
        "--base_omega0", type=int, default=0, help="SIREN Omega 0 parameter for the base model."
    )
    parser.add_argument(
        "--base_omegaW",
        type=int,
        default=0,
        help="SIREN Omega 0 parameter for hidden layers of the base model.",
    )
    parser.add_argument(
        "--residual_omega0", type=int, default=0, help="SIREN Omega 0 parameter for the residual model."
    )
    parser.add_argument(
        "--residual_omegaW",
        type=int,
        default=0,
        help="SIREN Omega 0 parameter for hidden layers for the residual model.",
    )
    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=0,
        help="# of points to fetch per iteration. By default, uses the # of mesh vertices.",
    )
    args = parser.parse_args()

    if not osp.exists(args.configpath):
        raise FileNotFoundError(
            f'Experiment configuration file "{args.configpath}" not found.'
        )

    if not osp.exists(args.meshpath):
        raise FileNotFoundError(f'Mesh file "{args.meshpath}" not found.')

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

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    devstr = args.device
    if "cuda" in devstr and not torch.cuda.is_available():
        devstr = "cpu"
        print("No CUDA available devices found on system. Using CPU.")

    device = torch.device(devstr)
    dataset = PointCloud(
        args.meshpath,
        batch_size=BATCH,
        use_curvature=False,
        device=device,
    )

    N = dataset.vertices.shape[0]
    nsteps = round(EPOCHS * (N / dataset.batch_size))
    warmup_steps = nsteps // 10
    print(f"Total # of training steps = {nsteps}")

    # Create the model and optimizer
    # Load coarse model
    basecfg = config["network"]["base"]
    base_model = SIREN(
        basecfg["in_coords"],
        basecfg["out_coords"],
        hidden_layer_config=basecfg["hidden_layers"],
        w0=basecfg["omega_0"] if not args.base_omega0 else args.base_omega0,
        ww=basecfg["omega_w"] if not args.base_omegaW else args.base_omegaW,
    ).to(device)
    print("==================== Base Model ====================")
    print(base_model)

    rescfg = config["network"]["residual"]
    res_model = SIREN(
        rescfg["in_coords"],
        rescfg["out_coords"],
        hidden_layer_config=rescfg["hidden_layers"],
        w0=rescfg["omega_0"] if not args.residual_omega0 else args.residual_omega0,
        ww=rescfg["omega_w"] if not args.residual_omegaW else args.residual_omegaW,
    ).to(device)
    print("==================== Residual Model ====================")
    print(res_model)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config["network"]["base"]["omega_0"] = base_model.w0
    config["network"]["base"]["omega_w"] = base_model.ww

    config["network"]["residual"]["omega_0"] = res_model.w0
    config["network"]["residual"]["omega_w"] = res_model.ww

    num_parameters_base = parameters_to_vector(base_model.parameters()).numel()
    num_parameters_res = parameters_to_vector(res_model.parameters()).numel()
    print("# parameters for base model =", num_parameters_base)
    print("# parameters for residual model =", num_parameters_res)

    optim = torch.optim.Adam(
        lr=config["optimizer"].get("lr", 1e-4),
        params=list(base_model.parameters()) + list(res_model.parameters())
    )
    training_loss = {}

    # TODO: CHANGE FROM HERE
    loss_fn = loss_add_detail_sitzmann(coarse_model)

    nonsurf = round(BATCH * 0.5)
    best_loss = torch.inf
    best_weights = None

    with open(osp.join(args.outputpath, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    start_training_time = time.time()
    for step in range(nsteps):
        samples = dataset[0]
        gt = samples[1]

        optim.zero_grad(set_to_none=True)
        y = model(samples[0]["coords"])
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

    # Save model states
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

    # Compute the file size of the best model file (in bytes)
    file_size = osp.getsize(best_model_path)
    file_size = file_size / 1024  # Convert bytes to kilobytes

    # Optionally, round the value to two decimal places:
    file_size = round(file_size, 2)

    # Write the training metrics to a CSV file in the output folder
    metrics_csv = osp.join(args.outputpath, "metrics.csv")
    with open(metrics_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header row
        csv_writer.writerow(
            ["Training Time (s)", "Num Parameters", "Model File Size (bytes)"]
        )
        # Write the metrics row
        csv_writer.writerow([training_time, num_parameters, file_size])

    print(f"Metrics saved to {metrics_csv}")
