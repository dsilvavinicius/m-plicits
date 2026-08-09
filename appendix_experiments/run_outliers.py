#!/usr/bin/env python
"""AbFK Q2: outlier geometry WITHOUT noise.

Contaminates the clean input with sparse uniform outlier points (0.1% / 1% of
N, uniform in the unit ball, random normals; seeded), keeps the mesh faces
(so the Eikonal/off-surface supervision still uses the clean raycasting
scene), trains the full nested-residual chain, and reconstructs.

Also produces the SPSR reconstruction of the SAME contaminated cloud for
comparison.

Run AFTER the main matrix: conda run -n new_i3d python run_outliers.py
Then: python run_metrics.py --meshes-dir <here>/outlier_meshes --out-prefix outlier_metrics
"""
import os
import os.path as osp
import subprocess
import sys

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement

HERE = osp.dirname(osp.abspath(__file__))
I3D = osp.dirname(HERE)
SCRIPTS = osp.join(I3D, "experiment_scripts")
CFG = osp.join(HERE, "configs")
OUT = osp.join(HERE, "outliers")
MESHES = osp.join(HERE, "outlier_meshes")

SHAPES = ["normalized_armadillo_gt", "normalized_lucy_gt"]
RATES = {"o01": 0.001, "o1": 0.01}
SEED = 668123


def contaminate(in_ply, out_ply, rate, seed):
    ply = PlyData.read(in_ply)
    vert = ply["vertex"].data
    n = vert.shape[0]
    k = max(1, int(round(rate * n)))
    rng = np.random.default_rng(seed)
    # uniform in the unit ball
    pts = rng.normal(size=(k, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= rng.uniform(0, 1, size=(k, 1)) ** (1 / 3)
    nrm = rng.normal(size=(k, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

    extra = np.zeros(k, dtype=vert.dtype)
    names = vert.dtype.names
    for i, ax in enumerate(("x", "y", "z")):
        extra[ax] = pts[:, i].astype(np.float32)
    for i, ax in enumerate(("nx", "ny", "nz")):
        if ax in names:
            extra[ax] = nrm[:, i].astype(np.float32)
    merged = np.concatenate([vert, extra])
    els = [PlyElement.describe(merged, "vertex")]
    if "face" in [e.name for e in ply.elements]:
        els.append(ply["face"])
    PlyData(els, text=False).write(out_ply)
    print(f"{osp.basename(out_ply)}: +{k} outliers ({rate:.1%} of {n})")


def run(cmd, cwd, log):
    os.makedirs(osp.dirname(log), exist_ok=True)
    print("RUN", " ".join(map(str, cmd)))
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=f,
                       stderr=subprocess.STDOUT, check=True)


def spsr(in_ply, out_ply, depth=10, trim_q=0.01):
    mesh_in = o3d.io.read_triangle_mesh(in_ply)
    pc = o3d.geometry.PointCloud()
    pc.points = mesh_in.vertices
    pc.normals = mesh_in.vertex_normals
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim_q))
    o3d.io.write_triangle_mesh(out_ply, mesh)


def main():
    sys.path.insert(0, I3D)
    sys.path.insert(0, HERE)
    from i3d.meshing import create_mesh_multistage
    from i3d.util import from_pth
    import evallib

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MESHES, exist_ok=True)
    py = sys.executable
    device = "cuda:0"
    for shape in SHAPES:
        clean = osp.join(I3D, "data", "normalized_sphere", "input", f"{shape}.ply")
        for tag, rate in RATES.items():
            data = osp.join(OUT, f"{shape}__{tag}.ply")
            if not osp.exists(data):
                contaminate(clean, data, rate, SEED)
            base = osp.join(OUT, shape, tag)
            cdir, mdir, fdir = (osp.join(base, s) for s in ("coarse", "medium", "fine"))
            if not osp.exists(osp.join(cdir, "best.pth")):
                run([py, osp.join(I3D, "train_sdf.py"), data, cdir,
                     osp.join(CFG, "coarse.yaml")], I3D, osp.join(base, "coarse.log"))
            if not osp.exists(osp.join(mdir, "best.pth")):
                run([py, osp.join(SCRIPTS, "train_sdf_on_neighborhood.py"), data,
                     mdir, osp.join(CFG, "medium_band_res.yaml"),
                     osp.join(cdir, "best.pth")], SCRIPTS, osp.join(base, "medium.log"))
            if not osp.exists(osp.join(fdir, "best.pth")):
                run([py, osp.join(SCRIPTS, "train_sdf_on_neighborhood_fine.py"),
                     data, fdir, osp.join(CFG, "fine_band_res.yaml"),
                     osp.join(cdir, "best.pth"), osp.join(mdir, "best.pth")],
                    SCRIPTS, osp.join(base, "fine.log"))

            v = np.asarray(o3d.io.read_triangle_mesh(data).vertices, dtype=np.float32)
            vt = torch.from_numpy(v).to(device)
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
            spsr_mesh = osp.join(MESHES, f"{shape}__{tag}__spsr.ply")
            if not osp.exists(spsr_mesh):
                spsr(data, spsr_mesh)
            print(f"{shape}/{tag}: done (d1={d1:.3e}, d2={d2:.3e})")
    print("done; run run_metrics.py --meshes-dir", MESHES,
          "--out-prefix outlier_metrics")


if __name__ == "__main__":
    main()
