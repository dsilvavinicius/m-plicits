#!/usr/bin/env python
"""Extract the adaptive band widths (Eq. 5) from the paper checkpoints.

For each Stanford shape:
  delta_1 = (1+eps) * max_j |f_1(x_j)|          (band for the medium residual)
  delta_2 = (1+eps) * max_j |(f_1+r_1)(x_j)|    (band for the fine residual)
with eps = 0.3 (the implementation's `* 1.3`), x_j = input point cloud.

This is the direct numerical answer to Reviewer AbFK's Eq. 5 concern: the max
runs over the INPUT SURFACE POINTS (where |f_i| is the residual fitting error,
tiny by construction), not over the domain.

Run: conda run -n new_i3d python extract_deltas.py
"""
import os.path as osp
import sys

import numpy as np
import open3d as o3d
import torch

HERE = osp.dirname(osp.abspath(__file__))
I3D = osp.dirname(HERE)
sys.path.insert(0, I3D)
from i3d.util import from_pth  # noqa: E402

SHAPES = [
    "normalized_armadillo_gt",
    "normalized_asian_dragon_gt",
    "normalized_lucy_gt",
    "normalized_thai_gt_half",
]
EPS = 0.3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def stage_dir(shape, stage):
    d = osp.join(I3D, "results", shape, stage)
    if osp.exists(osp.join(d, "best.pth")):
        return d
    return None


@torch.no_grad()
def max_abs(models, pts):
    out = 0.0
    for i in range(0, pts.shape[0], 2_000_000):
        chunk = pts[i : i + 2_000_000]
        s = torch.zeros(chunk.shape[0], 1, device=DEVICE)
        for m in models:
            s += m(chunk)["model_out"]
        out = max(out, float(s.abs().max()))
    return out


def main():
    rows = []
    for shape in SHAPES:
        ply = osp.join(I3D, "data", "normalized_sphere", "input", f"{shape}.ply")
        cd, md = stage_dir(shape, "coarse"), stage_dir(shape, "medium")
        if not cd or not md:
            print(f"[skip] {shape}: missing checkpoints")
            continue
        v = np.asarray(o3d.io.read_triangle_mesh(ply).vertices, dtype=np.float32)
        pts = torch.from_numpy(v).to(DEVICE)
        coarse = from_pth(osp.join(cd, "best.pth"), w0=1).eval().to(DEVICE)
        medium = from_pth(osp.join(md, "best.pth"), w0=1).eval().to(DEVICE)
        m1 = max_abs([coarse], pts)
        m2 = max_abs([coarse, medium], pts)
        rows.append((shape, v.shape[0], m1, (1 + EPS) * m1, m2, (1 + EPS) * m2))
        print(f"{shape}: n={v.shape[0]}  max|f1|={m1:.3e} d1={(1+EPS)*m1:.3e}  "
              f"max|f1+r1|={m2:.3e} d2={(1+EPS)*m2:.3e}")

    out = osp.join(HERE, "deltas_table.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Adaptive band widths (Eq. 5, eps=0.3) from paper checkpoints\n\n")
        f.write("Domain: unit sphere (diameter 2). The max runs over the input"
                " surface points x_j, NOT the domain.\n\n")
        f.write("| Shape | #points | max_j abs(f1(x_j)) | delta_1 | max_j abs((f1+r1)(x_j)) | delta_2 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]:.3e} | {r[3]:.3e} | {r[4]:.3e} | {r[5]:.3e} |\n")
    print(f"table -> {out}")


if __name__ == "__main__":
    main()
