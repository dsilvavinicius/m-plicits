#!/usr/bin/env python
"""Export a released checkpoint set to the real-time renderer.

Converts results/<shape>/{coarse,medium,fine}/best.pth into the renderer's
flat float32 .bin pairs and writes a runtime experiment descriptor, so any
released model renders without a registry entry in state.cu:

    python renderer/scripts/export_experiment.py results/normalized_lucy_gt \
        --out renderer/cuda/build/Release/data/released
    cd renderer/cuda/build/Release
    MIP-plicitsRenderer.exe -experiment_file=data/released/normalized_lucy_gt.exp -iters=20,5,5 -delta=0.02 -normal_lod=2

The released checkpoints store the SIREN frequency baked into the weights
(the training code saves them with omega_0 = 1), which is why the descriptor
uses surface_w0 = 1; pass --w0 for checkpoints saved otherwise. Use --flip-y
if the model renders upside down (its native y axis points down).
"""
import argparse
import os
import os.path as osp
import struct

import torch

STAGES = ("coarse", "medium", "fine")


def flatten_state_dict(sd):
    """Row-major weights and biases in layer order, the renderer's layout."""
    tensors = list(sd.values())
    weights, biases = [], []
    for i in range(0, len(tensors), 2):
        weights.append(tensors[i].detach().cpu().float().numpy().reshape(-1))
        biases.append(tensors[i + 1].detach().cpu().float().numpy().reshape(-1))
    hidden = tensors[0].shape[0]
    n_hidden_gemms = sum(1 for t in tensors[2:-2:2] if tuple(t.shape) == (hidden, hidden))
    return weights, biases, int(hidden), n_hidden_gemms


def write_bin(path, arrays):
    flat = [float(x) for a in arrays for x in a]
    with open(path, "wb") as f:
        f.write(struct.pack("f" * len(flat), *flat))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="results/<shape> with coarse/, medium/, fine/ subfolders")
    ap.add_argument("--out", required=True, help="output folder (under the renderer's data root)")
    ap.add_argument("--name", default=None, help="descriptor / file stem (default: results folder name)")
    ap.add_argument("--data-root", default="data", help="path prefix the renderer will use (default: data)")
    ap.add_argument("--w0", type=int, default=1, help="surface_W0 for the renderer's GEMM kernels")
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--swap-y-and-z", action="store_true")
    ap.add_argument("--invert-z", action="store_true")
    args = ap.parse_args()

    name = args.name or osp.basename(osp.normpath(args.results_dir)).replace(".ply", "")
    os.makedirs(args.out, exist_ok=True)
    rel_dir = osp.basename(osp.normpath(args.out))
    lines = [f"# exported from {osp.abspath(args.results_dir)}"]
    for lod, stage in enumerate(STAGES):
        ck = osp.join(args.results_dir, stage, "best.pth")
        if not osp.exists(ck):
            raise SystemExit(f"missing {ck}")
        sd = torch.load(ck, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        w, b, hidden, n_gemms = flatten_state_dict(sd)
        stem = f"{name}_{stage}"
        write_bin(osp.join(args.out, stem + "_weights.bin"), w)
        write_bin(osp.join(args.out, stem + "_biases.bin"), b)
        lines += [f"lod{lod}_layers = {n_gemms}", f"lod{lod}_hidden = {hidden}",
                  f"lod{lod}_weights = {args.data_root}/{rel_dir}/{stem}_weights.bin",
                  f"lod{lod}_biases = {args.data_root}/{rel_dir}/{stem}_biases.bin"]
        print(f"{stage}: width {hidden}, {n_gemms} hidden GEMM(s), {sum(len(x) for x in w)} weights")
    lines += ["textures_layers = 0", "is3d = 1", f"surface_w0 = {args.w0}", "textures_w0 = 0",
              f"swap_y_and_z = {int(args.swap_y_and_z)}", f"invert_z = {int(args.invert_z)}",
              f"flip_y = {int(args.flip_y)}"]
    desc = osp.join(args.out, name + ".exp")
    with open(desc, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("descriptor:", desc)


if __name__ == "__main__":
    main()
