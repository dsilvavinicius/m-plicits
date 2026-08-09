import glob, json, os.path as osp
import numpy as np, open3d as o3d
rec = {}
for p in sorted(glob.glob("meshes_nocull/*.ply")):
    base = osp.basename(p)[:-4]
    shape, cond, tag = base.split("__")
    m = o3d.io.read_triangle_mesh(p)
    _, cnt, _ = m.cluster_connected_triangles()
    cnt = np.asarray(cnt)
    bb = m.get_axis_aligned_bounding_box()
    rec.setdefault(f"{shape}__{cond}", {})[tag] = {
        "components": int(len(cnt)),
        "bbox_diag": float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound())),
    }
json.dump(rec, open("nocull_structure.json", "w"), indent=2)
print(f"{'unit':34s} {'aNC':>14s} {'cNC':>16s}")
for k in sorted(rec):
    a = rec[k].get("aNC"); c = rec[k].get("cNC")
    if a and c:
        print(f"{k:34s} {a['components']:5d}/{a['bbox_diag']:.2f}   {c['components']:6d}/{c['bbox_diag']:.2f}")
