#!/usr/bin/env python
# coding: utf-8

"""
Simple script to run multiple SDF reconstructions given a base log directory
and a set of checkpoints.
"""

import argparse
import os
import os.path as osp
import numpy as np
import torch
from i3d.meshing import create_mesh_multistage
from i3d.util import from_pth


def adaptive_deltas(models, input_ply, device, eps=0.3, batch=2 ** 18):
    """Band widths of the paper's inference (Eq. 5): delta_i = (1+eps) * max_j |f_i(x_j)|,
    where f_i is the partial sum up to level i and x_j runs over the training points.
    `models` is ordered coarse -> fine; returns one delta per residual level."""
    import open3d as o3d
    pts = torch.from_numpy(np.asarray(o3d.io.read_point_cloud(input_ply).points)).float().to(device)
    deltas = []
    with torch.no_grad():
        for level in range(1, len(models)):
            worst = 0.0
            for i in range(0, pts.shape[0], batch):
                chunk = pts[i:i + batch]
                s = sum(models[k](chunk)["model_out"].squeeze(-1) for k in range(level))
                worst = max(worst, float(s.abs().max()))
            deltas.append((1.0 + eps) * worst)
    return deltas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run marching cubes using a trained model."
    )
    parser.add_argument(
        "model_path",
        help="Path to the PyTorch weights file"
    )
    parser.add_argument(
        "--coarse_path", default=None,
        help="Path to the PyTorch weights file for the coarse network"
    )
    parser.add_argument(
        "--medium_path", default=None, help="Path to the medium resolution model. Only use"
        " this when reconstructing a fine-level model."
    )
    parser.add_argument(
        "output_path",
        help="Path to the output mesh file"
    )
    parser.add_argument(
        "--w0", type=int, default=1,
        help="Value for \\omega_0. Default is 1."
    )
    parser.add_argument(
        "--resolution", "-r", default=128, type=int,
        help="Resolution to use on marching cubes. Default is 128."
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to run the inference. By default is \"cpu\"."
    )
    parser.add_argument(
        "--multistage", action="store_true", help="Use the multiscale optimization (finer levels only evaluated near the surface)."
    )
    parser.add_argument(
        "--input", default=None,
        help="Training point cloud (.ply). With --multistage, the band widths are the paper's adaptive"
             " deltas computed on it (Eq. 5: delta_i = 1.3 * max_j |f_i(x_j)|); without it, fixed"
             " fallback widths are used, which is NOT the paper's inference."
    )
    parser.add_argument(
        "--deltas", type=float, nargs="+", default=None,
        help="Explicit band widths (one per residual level), overriding --input / the fallback."
    )

    args = parser.parse_args()
    out_dir = osp.split(args.output_path)[0]
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir)

    device = args.device
    if "cuda" in args.device and not torch.cuda.is_available():
        print(f"[WARNING] Selecte \"{device}\" for inference is not available. Using CPU.")
        device = "cpu"

    device = torch.device(device)
    model = from_pth(args.model_path, w0=args.w0).eval().to(device)
    print(model)
    model_list = [model]

    print(f"Running marching cubes running with resolution {args.resolution}")
    if args.medium_path is not None:
        print("Using residual approach for fine level")
        model_medium = from_pth(args.medium_path, w0=1).eval().to(device)
        model_list.append(model_medium)
        print("========== Medium model ==========")
        print(model_medium)
    if args.coarse_path is not None:
        print("Using the residual approach")
        model_coarse = from_pth(args.coarse_path, w0=1).eval().to(device)
        model_list.append(model_coarse)
        print("========== Coarse model ==========")
        print(model_coarse)

    if args.multistage:
        if args.deltas is not None:
            deltas = list(args.deltas)
        elif args.input is not None:
            deltas = adaptive_deltas(model_list[::-1], args.input, device)
            print("adaptive band widths (Eq. 5):", ", ".join(f"{d:.3e}" for d in deltas))
        else:
            deltas = [0.1, 0.06][:len(model_list) - 1]
            print("[WARNING] --multistage without --input: using fixed band widths"
                  f" {deltas}; pass the training point cloud for the paper's adaptive bands.")
    else:
        deltas=[]

    _, _, _, _, sampling_time = create_mesh_multistage(
        model_list[::-1],
        deltas=deltas,
        filename=args.output_path,
        N=args.resolution,
        max_batch=32**3,
        device=device
    )

    print(f"SAMPLING_TIME_S:{sampling_time}")
    print("Done")
