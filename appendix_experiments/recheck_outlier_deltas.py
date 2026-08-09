"""Recompute delta_1 on the outlier-contaminated inputs (Eq. 5, eps=0.3).

Verifies the rebuttal claim that a handful of stray points inflates the band:
delta_1 = 1.3 * max_j |f_1(x_j)| over the CONTAMINATED point set, so a single
outlier far from the surface drags the max up and the band degenerates.
"""
import os.path as osp
import sys

import numpy as np
import open3d as o3d
import torch

HERE = osp.dirname(osp.abspath(__file__))
I3D = osp.dirname(HERE)
sys.path.insert(0, I3D)
sys.path.insert(0, HERE)
import evallib  # noqa: E402
from i3d.util import from_pth  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

for shape in ("normalized_armadillo_gt", "normalized_lucy_gt"):
    cdir = osp.join(I3D, "results", shape, "coarse", "best.pth")
    coarse = from_pth(cdir, w0=1).eval().to(DEV)
    clean = osp.join(I3D, "data", "normalized_sphere", "input", f"{shape}.ply")
    for tag, path in [("clean", clean)] + [
            (t, osp.join(HERE, "outliers", f"{shape}__{t}.ply"))
            for t in ("o01", "o1")]:
        if not osp.exists(path):
            print(f"{shape:26s} {tag:5s} MISSING {path}")
            continue
        pcd = o3d.io.read_point_cloud(path)
        v = np.asarray(pcd.points, dtype=np.float32)
        if v.shape[0] == 0:
            v = np.asarray(o3d.io.read_triangle_mesh(path).vertices,
                           dtype=np.float32)
        vt = torch.from_numpy(v).to(DEV)
        m = evallib.max_abs_at_points([coarse], vt, device=DEV)
        print(f"{shape:26s} {tag:5s} n={v.shape[0]:>9,d} "
              f"max|f1|={m:.4e}  delta1={1.3 * m:.4e}", flush=True)
    del coarse
    torch.cuda.empty_cache()
