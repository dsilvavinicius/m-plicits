"""Repair the 4 degenerate (zero-length) vertex normals in thai_half.

Why: i3d's calc_deltas normalizes each vertex normal by its length; a
zero-length normal yields inf/NaN, which propagates into the training batch
and NaNs the medium-stage loss (best_weights is then never set and the run
crashes at save time). This is why the paper's own results/
normalized_thai_gt_half/medium is incomplete and fine/ is empty.

Minimal intervention: only the degenerate normals are replaced (with the
area-weighted normal recomputed by Open3D; radial direction as fallback).
All other 2,499,992 normals and every vertex position are untouched.
Originals are kept as <file>.orig.

Run on the server:  python fix_thai_normals.py <ply> [<ply> ...]
"""
import os
import os.path as osp
import shutil
import sys

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement


def fix(path):
    ply = PlyData.read(path)
    vert = ply["vertex"].data
    n = np.stack([np.asarray(vert[a]) for a in ("nx", "ny", "nz")], 1).astype(np.float64)
    L = np.linalg.norm(n, axis=1)
    bad = np.where(L < 1e-8)[0]
    if bad.size == 0:
        print(f"{osp.basename(path)}: no degenerate normals, untouched")
        return
    xyz = np.stack([np.asarray(vert[a]) for a in ("x", "y", "z")], 1).astype(np.float64)

    # Recompute normals with Open3D and use them for the degenerate vertices.
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    rec = np.asarray(mesh.vertex_normals)
    repl = np.zeros((bad.size, 3))
    for i, vi in enumerate(bad):
        cand = rec[vi] if vi < rec.shape[0] else np.zeros(3)
        if np.linalg.norm(cand) < 1e-8:            # fallback: radial
            cand = xyz[vi]
            if np.linalg.norm(cand) < 1e-8:        # last resort
                cand = np.array([0.0, 0.0, 1.0])
        repl[i] = cand / np.linalg.norm(cand)

    if not osp.exists(path + ".orig"):
        shutil.copy2(path, path + ".orig")
    out = vert.copy()
    for i, vi in enumerate(bad):
        out["nx"][vi], out["ny"][vi], out["nz"][vi] = repl[i].astype(np.float32)
    els = [PlyElement.describe(out, "vertex")]
    if "face" in [e.name for e in ply.elements]:
        els.append(ply["face"])
    PlyData(els, text=False).write(path)
    print(f"{osp.basename(path)}: repaired {bad.size} degenerate normals "
          f"(indices {bad[:8].tolist()}), backup at <file>.orig")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        fix(p)
