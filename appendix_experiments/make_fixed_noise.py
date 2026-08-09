"""Regenerate CORRECT 1%-noise inputs for asian_dragon and thai_half.

Verification (check_noise_nn.py) showed the canonical noise_data files for
these two shapes carry no real displacement (NN-to-clean ~1e-4) while
armadillo/lucy carry genuine ~1% noise (~1e-2). This script produces proper
perturbations with the established protocol: displace each vertex along its
GT normal by U(-d, d), d = 1% of the bbox diagonal; normals and faces kept.
Seeded. Output goes to rebuttal_2026/fixed_noise/ (canonical files untouched).
"""
import os
import os.path as osp

import numpy as np
import open3d as o3d

HERE = osp.dirname(osp.abspath(__file__))
I3D = osp.dirname(HERE)
OUT = osp.join(HERE, "fixed_noise")
SHAPES = ["normalized_asian_dragon_gt", "normalized_thai_gt_half"]
DELTA_FRAC = 0.01
SEED = 668123

os.makedirs(OUT, exist_ok=True)
for s in SHAPES:
    src = osp.join(I3D, "data", "normalized_sphere", "input", f"{s}.ply")
    dst = osp.join(OUT, f"{s}.ply")
    mesh = o3d.io.read_triangle_mesh(src)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    n = np.asarray(mesh.vertex_normals)
    aabb = mesh.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound()))
    d = DELTA_FRAC * diag
    rng = np.random.default_rng(SEED)
    noise = rng.uniform(-d, d, size=(v.shape[0], 1))
    mesh.vertices = o3d.utility.Vector3dVector(v + noise * n)
    mesh.vertex_normals = o3d.utility.Vector3dVector(n)  # keep GT normals
    o3d.io.write_triangle_mesh(dst, mesh)
    print(f"{s}: n={v.shape[0]} diag={diag:.4f} d={d:.5f} -> {dst}")
print("done")
