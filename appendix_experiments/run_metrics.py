#!/usr/bin/env python
"""Compute CD/Hausdorff/IoU for all rebuttal meshes using the PAPER's metric
code (J:/projects/metrics/meshes/compute_distance_metrics.py).

Builds mesh_pairs.json (reconstruction vs CLEAN ground-truth input) from
meshes/<shape>__<cond>__<cell>.ply, then invokes the paper script inside the
`metrics` conda env with the paper protocol (500K samples, L2, IoU voxel 0.01).

Run from any env:  python run_metrics.py
"""
import argparse
import glob
import json
import os.path as osp
import subprocess

HERE = osp.dirname(osp.abspath(__file__))
I3D = osp.dirname(HERE)
METRICS_REPO = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "metrics")

NUM_POINTS = 500_000
IOU_VOXEL = 0.01
NORM = 2


def gt_clean_ply(shape):
    return osp.join(I3D, "data", "normalized_sphere", "input", f"{shape}.ply")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_points", type=int, default=NUM_POINTS)
    ap.add_argument("--meshes-dir", default=osp.join(HERE, "meshes"))
    ap.add_argument("--out-prefix", default="paper_metrics")
    args = ap.parse_args()

    meshes = sorted(glob.glob(osp.join(args.meshes_dir, "*__*__*.ply")))
    pairs = []
    skipped = []
    for m in meshes:
        shape = osp.basename(m).split("__")[0]
        gt = gt_clean_ply(shape)
        if not osp.exists(gt):
            skipped.append(m)
            continue
        pairs.append({"mesh1": m, "mesh2": gt})
    if skipped:
        print(f"[warn] {len(skipped)} meshes skipped (no GT): {skipped}")
    if not pairs:
        print("No meshes found.")
        return

    pairs_json = osp.join(HERE, f"{args.out_prefix}_pairs.json")
    with open(pairs_json, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
    print(f"{len(pairs)} pairs -> {pairs_json}")

    csv_out = osp.join(HERE, f"{args.out_prefix}.csv")
    tex_out = osp.join(HERE, f"{args.out_prefix}.tex")
    cmd = [
        "conda", "run", "-n", "metrics", "--no-capture-output", "python",
        osp.join(METRICS_REPO, "meshes", "compute_distance_metrics.py"),
        "--json_file", pairs_json,
        "--csv_file", csv_out,
        "--latex_file", tex_out,
        "--num_points", str(args.num_points),
        "--norm", str(NORM),
        "--iou_voxel_size", str(IOU_VOXEL),
    ]
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"paper metrics -> {csv_out}")


if __name__ == "__main__":
    main()
