#!/usr/bin/env python
"""AbFK stage-robustness experiment: under-train f1, keep the pipeline fixed.

Trains the coarse level for 25% / 50% / 100% of its epochs, then runs the
unchanged medium+fine nested-residual stages on top of each variant.
Punchline: delta_1 widens automatically (Eq. 5) and the final CD stays put.

Run AFTER the main matrix (GPU): conda run -n new_i3d python run_undertrained.py
Then: python run_metrics.py --meshes-dir <here>/undertrained_meshes --out-prefix undertrained_metrics
"""
import os
import os.path as osp
import subprocess
import sys

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
I3D = osp.dirname(HERE)
SCRIPTS = osp.join(I3D, "experiment_scripts")
CFG = osp.join(HERE, "configs")
OUT = osp.join(HERE, "undertrained")
MESHES = osp.join(HERE, "undertrained_meshes")

SHAPES = ["normalized_armadillo_gt", "normalized_lucy_gt"]
FRACTIONS = {"c25": 875, "c50": 1750, "c100": 3500}
COARSE_EPOCHS_FULL = 3500


def run(cmd, cwd, log):
    os.makedirs(osp.dirname(log), exist_ok=True)
    print("RUN", " ".join(map(str, cmd)))
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=f,
                       stderr=subprocess.STDOUT, check=True)


def main():
    sys.path.insert(0, I3D)
    sys.path.insert(0, HERE)
    import open3d as o3d
    from i3d.meshing import create_mesh_multistage
    from i3d.util import from_pth
    import evallib

    os.makedirs(MESHES, exist_ok=True)
    py = sys.executable
    device = "cuda:0"
    rows = []
    for shape in SHAPES:
        mesh = osp.join(I3D, "data", "normalized_sphere", "input", f"{shape}.ply")
        v = np.asarray(o3d.io.read_triangle_mesh(mesh).vertices, dtype=np.float32)
        vt = torch.from_numpy(v).to(device)
        for tag, ep in FRACTIONS.items():
            base = osp.join(OUT, shape, tag)
            cdir, mdir, fdir = (osp.join(base, s) for s in ("coarse", "medium", "fine"))
            if not osp.exists(osp.join(cdir, "best.pth")):
                run([py, osp.join(I3D, "train_sdf.py"), mesh, cdir,
                     osp.join(CFG, "coarse.yaml"), "--nepochs", ep],
                    I3D, osp.join(base, "coarse.log"))
            if not osp.exists(osp.join(mdir, "best.pth")):
                run([py, osp.join(SCRIPTS, "train_sdf_on_neighborhood.py"), mesh,
                     mdir, osp.join(CFG, "medium_band_res.yaml"),
                     osp.join(cdir, "best.pth")],
                    SCRIPTS, osp.join(base, "medium.log"))
            if not osp.exists(osp.join(fdir, "best.pth")):
                run([py, osp.join(SCRIPTS, "train_sdf_on_neighborhood_fine.py"),
                     mesh, fdir, osp.join(CFG, "fine_band_res.yaml"),
                     osp.join(cdir, "best.pth"), osp.join(mdir, "best.pth")],
                    SCRIPTS, osp.join(base, "fine.log"))

            c = from_pth(osp.join(cdir, "best.pth"), w0=1).eval().to(device)
            md = from_pth(osp.join(mdir, "best.pth"), w0=1).eval().to(device)
            fn = from_pth(osp.join(fdir, "best.pth"), w0=1).eval().to(device)
            d1 = 1.3 * evallib.max_abs_at_points([c], vt, device=device)
            d2 = 1.3 * evallib.max_abs_at_points([c, md], vt, device=device)
            out_mesh = osp.join(MESHES, f"{shape}__{tag}__a.ply")
            if not osp.exists(out_mesh):
                pad = 2 * (2.0 / 511)
                create_mesh_multistage([c, md, fn], deltas=[d1 + pad, d2 + pad],
                                       filename=out_mesh, N=512,
                                       max_batch=32**3, device=device)
            rows.append((shape, tag, ep, d1, d2))
            print(f"{shape}/{tag}: coarse epochs={ep} d1={d1:.3e} d2={d2:.3e}")

    with open(osp.join(HERE, "undertrained_deltas.md"), "w", encoding="utf-8") as f:
        f.write("| Shape | Coarse epochs | delta_1 | delta_2 |\n|---|---|---|---|\n")
        for s, tag, ep, d1, d2 in rows:
            f.write(f"| {s} | {ep} | {d1:.3e} | {d2:.3e} |\n")
    print("done; run run_metrics.py --meshes-dir", MESHES,
          "--out-prefix undertrained_metrics")


if __name__ == "__main__":
    main()
