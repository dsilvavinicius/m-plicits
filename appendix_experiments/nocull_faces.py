"""Face/vertex counts for the unculled meshes, and a Chamfer that works on
meshes too large for PyTorch3D's area-weighted sampler (>2^24 faces).

Sampling backend is Open3D's area-uniform sampler; the Chamfer itself is the
same squared-L2 bidirectional mean used by the paper's script, so the values
are directly comparable to our other tables.
"""
import glob
import json
import os.path as osp

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

I3D = osp.dirname(osp.dirname(osp.abspath(__file__)))
N = 500_000


def gt_path(shape):
    return osp.join(I3D, "data", "normalized_sphere", "input", f"{shape}.ply")


def sample(path, n, seed=0):
    m = o3d.io.read_triangle_mesh(path)
    o3d.utility.random.seed(seed)
    pts = np.asarray(m.sample_points_uniformly(n).points)
    return pts, len(m.vertices), len(m.triangles)


rows = []
for p in sorted(glob.glob("meshes_nocull/*.ply")):
    shape, cond, tag = osp.basename(p)[:-4].split("__")
    a, nv, nf = sample(p, N)
    b, _, _ = sample(gt_path(shape), N, seed=1)
    d_ab, _ = cKDTree(b).query(a, k=1, workers=-1)
    d_ba, _ = cKDTree(a).query(b, k=1, workers=-1)
    cd = float((d_ab ** 2).mean() + (d_ba ** 2).mean())
    rows.append(dict(shape=shape, cond=cond, tag=tag, verts=nv, faces=nf, cd=cd))
    print(f"{shape[:24]:24s} {cond:5s} {tag:4s} faces={nf:>9,d} CD={cd:.3e}",
          flush=True)

json.dump(rows, open("nocull_faces.json", "w"), indent=2)

print(f"\n=== means over {len(rows) // 2} units ===")
for tag in ("aNC", "cNC"):
    sel = [r for r in rows if r["tag"] == tag]
    if sel:
        print(f"{tag}: mean faces={np.mean([r['faces'] for r in sel]):>12,.0f}  "
              f"mean CD={np.mean([r['cd'] for r in sel]):.3e}  "
              f"median CD={np.median([r['cd'] for r in sel]):.3e}")
