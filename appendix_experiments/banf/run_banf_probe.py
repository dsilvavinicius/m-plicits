"""BANF run with PER-LEVEL extraction + immediate quick evaluation.

Purpose: get early evidence that the corrected implementation is competitive,
without waiting for the full run. After each level k finishes we extract the
partial recomposition (levels 0..k) and immediately score it, so the numbers
can be compared directly against BANF's own per-resolution table
(Asian Dragon 1.23 / 0.65 / 0.32 and Thai 1.41 / 0.68 / 0.40, x10^-2,
mean Euclidean Chamfer).

The score printed here is a MONITORING metric computed with Open3D KD-trees
(no pytorch3d on the server). It reports both conventions:
  cd_sq   - bidirectional mean SQUARED distance   (our tables' convention)
  cd_mean - bidirectional mean EUCLIDEAN distance (BANF's convention, x10^-2)
Final numbers for the rebuttal are recomputed locally with the paper's script.

Usage: python run_banf_probe.py --shape <name> --cond clean|noise
"""
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
import open3d as o3d
import torch

HERE = osp.dirname(osp.abspath(__file__))
REB = osp.dirname(HERE)
I3D = osp.dirname(REB)
sys.path.insert(0, HERE)
from banf_sdf import train_banf, extract_mesh  # noqa: E402

# BANF's published Chamfer-L2 (x10^-2, mean Euclidean) at 32/64/128
PUBLISHED = {
    "normalized_asian_dragon_gt": [1.23, 0.65, 0.32],
    "normalized_thai_gt_half": [1.41, 0.68, 0.40],
}
N_EVAL = 200_000


def input_ply(shape, cond):
    if cond == "clean":
        return osp.join(I3D, "data", "normalized_sphere", "input", f"{shape}.ply")
    return osp.join(I3D, "data", "normalized_noise", "noise_data", f"{shape}.ply")


def sample(path, n, seed=0):
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles) == 0:
        return np.asarray(m.vertices, dtype=np.float64)
    o3d.utility.random.seed(seed)
    return np.asarray(m.sample_points_uniformly(n).points, dtype=np.float64)


def quick_cd(recon_path, gt_path):
    a, b = sample(recon_path, N_EVAL), sample(gt_path, N_EVAL, seed=1)
    pa, pb = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pa.points = o3d.utility.Vector3dVector(a)
    pb.points = o3d.utility.Vector3dVector(b)
    d_ab = np.asarray(pa.compute_point_cloud_distance(pb))
    d_ba = np.asarray(pb.compute_point_cloud_distance(pa))
    cd_sq = float((d_ab ** 2).mean() + (d_ba ** 2).mean())
    cd_mean = float(0.5 * (d_ab.mean() + d_ba.mean()))
    return cd_sq, cd_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--cond", default="clean", choices=["clean", "noise"])
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()

    src = input_ply(args.shape, args.cond)
    gt = input_ply(args.shape, "clean")          # always score against clean GT
    outdir = osp.join(REB, "meshes")
    os.makedirs(outdir, exist_ok=True)
    probe_csv = osp.join(HERE, "banf_probe.csv")
    if not osp.exists(probe_csv):
        with open(probe_csv, "w", encoding="utf-8") as f:
            f.write("shape,cond,level,lattice,cd_sq,cd_mean_x100,published_x100\n")

    pub = PUBLISHED.get(args.shape, [None, None, None])
    t0 = time.time()

    def on_level(cascade, k):
        lattice = [32, 64, 128, 256][k]
        path = osp.join(outdir, f"{args.shape}__{args.cond}__banfL{k}.ply")
        extract_mesh(cascade, path, N=args.res, upto=k + 1)
        try:
            cd_sq, cd_mean = quick_cd(path, gt)
        except Exception as e:                      # empty mesh, etc.
            print(f"[probe] level {k}: eval failed ({e})", flush=True)
            return
        p = pub[k] if k < len(pub) else None
        msg = (f"[probe] {args.shape}/{args.cond} level {k} (lattice {lattice}): "
               f"cd_sq={cd_sq:.3e}  cd_mean={cd_mean*100:.2f}e-2")
        if p:
            msg += f"  | BANF published {p:.2f}e-2  -> ratio {cd_mean*100/p:.2f}x"
        msg += f"  [{(time.time()-t0)/60:.0f} min]"
        print(msg, flush=True)
        with open(probe_csv, "a", encoding="utf-8") as f:
            f.write(f"{args.shape},{args.cond},{k},{lattice},{cd_sq:.6e},"
                    f"{cd_mean*100:.4f},{p if p else ''}\n")

    cascade = train_banf(src, on_level=on_level)
    final = osp.join(outdir, f"{args.shape}__{args.cond}__banf.ply")
    extract_mesh(cascade, final, N=args.res)
    coarse = osp.join(outdir, f"{args.shape}__{args.cond}__banfcoarse.ply")
    extract_mesh(cascade, coarse, N=args.res, upto=1)
    print(f"DONE {args.shape}/{args.cond} in {(time.time()-t0)/60:.0f} min",
          flush=True)


if __name__ == "__main__":
    main()
