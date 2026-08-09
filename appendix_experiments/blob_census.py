"""Census of spurious off-surface geometry across all ablation meshes.

For every (shape, cond, cell in a/b/c): connected components and the maximum
sample distance to GT (the blob detector Hausdorff approximates). Uses the
retrained clean-(a) meshes where they exist.

Purpose: replace the artifact-sensitive 'IoU<0.35 failure' counts with a
protocol-neutral, directly verifiable statement: which reconstructions carry
far off-surface components, and how far.
"""
import os.path as osp

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

I3D = osp.dirname(osp.dirname(osp.abspath(__file__)))
N = 200_000
TEN = ["normalized_armadillo_gt", "normalized_asian_dragon_gt",
       "normalized_lucy_gt", "normalized_thai_gt_half",
       "44234", "64764", "68381", "72870", "73075", "77245"]


def gt_tree(shape):
    m = o3d.io.read_triangle_mesh(
        osp.join(I3D, "data", "normalized_sphere", "input", f"{shape}.ply"))
    return cKDTree(np.asarray(m.sample_points_uniformly(N).points))


def mesh_path(shape, cond, cell):
    if cell == "a" and cond == "clean":
        p = osp.join("meshes_retrain", f"{shape}__clean__a.ply")
        if osp.exists(p):
            return p
    return osp.join("meshes", f"{shape}__{cond}__{cell}.ply")


print(f"{'shape':26s} {'cond':6s} " +
      " ".join(f"{c+':comps/maxdev':>18s}" for c in "abc"))
summary = {}
for shape in TEN:
    tree = gt_tree(shape)
    for cond in ("clean", "noise"):
        row = f"{shape:26s} {cond:6s} "
        for cell in "abc":
            p = mesh_path(shape, cond, cell)
            if not osp.exists(p):
                row += f"{'MISSING':>18s} "
                continue
            m = o3d.io.read_triangle_mesh(p)
            _, cnt, _ = m.cluster_connected_triangles()
            pts = np.asarray(m.sample_points_uniformly(N).points)
            d, _ = tree.query(pts, k=1, workers=-1)
            mx = float(d.max())
            summary.setdefault((cond, cell), []).append((shape, len(cnt), mx))
            row += f"{len(np.asarray(cnt)):>6d}/{mx:>10.2e} "
        print(row, flush=True)

print("\n=== shapes with far off-surface geometry (maxdev > 5e-3) ===")
for cond in ("clean", "noise"):
    for cell in "abc":
        rows = summary.get((cond, cell), [])
        bad = [(s, mx) for s, _, mx in rows if mx > 5e-3]
        names = ", ".join(f"{s.replace('normalized_','').replace('_gt','')}"
                          f"({mx:.1e})" for s, mx in sorted(bad, key=lambda x: -x[1]))
        print(f"{cond:6s} ({cell}): {len(bad)}/10  {names}")
