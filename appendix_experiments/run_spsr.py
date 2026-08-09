#!/usr/bin/env python
"""Screened Poisson Surface Reconstruction baseline (Reviewer Vjd5).

Open3D SPSR (depth 10) on the clean and noisy inputs, with the standard
low-density-vertex trim (1% quantile). Produces meshes/<shape>__<cond>__spsr.ply;
CD/Hausdorff/IoU are then computed by run_metrics.py with the paper's metric
code.

Run: conda run -n new_i3d python run_spsr.py
"""
import os
import os.path as osp
import time

import numpy as np
import open3d as o3d

HERE = osp.dirname(osp.abspath(__file__))
I3D = osp.dirname(HERE)

SHAPES = [
    "normalized_armadillo_gt",
    "normalized_asian_dragon_gt",
    "normalized_lucy_gt",
    "normalized_thai_gt_half",
    "44234", "64764", "68381", "72870", "73075", "77245",
]
CONDS = ["clean", "noise"]
DEPTH = 10
TRIM_QUANTILE = 0.01


def input_ply(shape, cond):
    if cond == "clean":
        return osp.join(I3D, "data", "normalized_sphere", "input", f"{shape}.ply")
    return osp.join(I3D, "data", "normalized_noise", "noise_data", f"{shape}.ply")


def spsr(in_ply, out_ply):
    mesh_in = o3d.io.read_triangle_mesh(in_ply)
    pc = o3d.geometry.PointCloud()
    pc.points = mesh_in.vertices
    if mesh_in.has_vertex_normals():
        pc.normals = mesh_in.vertex_normals
    else:
        pc.estimate_normals()
    t0 = time.time()
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=DEPTH)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, TRIM_QUANTILE))
    dt = time.time() - t0
    o3d.io.write_triangle_mesh(out_ply, mesh)
    return dt


def main():
    outdir = osp.join(HERE, "meshes")
    os.makedirs(outdir, exist_ok=True)
    times = []
    for shape in SHAPES:
        for cond in CONDS:
            src = input_ply(shape, cond)
            if not osp.exists(src):
                print(f"[skip] {src} missing")
                continue
            out = osp.join(outdir, f"{shape}__{cond}__spsr.ply")
            if not osp.exists(out):
                dt = spsr(src, out)
                times.append((shape, cond, dt))
                print(f"{shape}/{cond}: SPSR depth {DEPTH} in {dt:.1f}s")

    with open(osp.join(HERE, "spsr_times.csv"), "w", encoding="utf-8") as f:
        f.write("shape,cond,seconds\n")
        for s, c, t in times:
            f.write(f"{s},{c},{t:.2f}\n")
    print("SPSR meshes done; run run_metrics.py for CD/Hausdorff/IoU.")


if __name__ == "__main__":
    main()
