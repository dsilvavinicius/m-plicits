"""Connected-component / bbox structure of the FINAL (corrected) BANF meshes.

Backs the rebuttal claim that BANF's finer levels convert input noise into
off-surface geometry on thin-structure shapes.
"""
import glob
import json
import os.path as osp

import numpy as np
import open3d as o3d

rows = []
for p in sorted(glob.glob("banf_final/*.ply")):
    m = o3d.io.read_triangle_mesh(p)
    _, cnt, _ = m.cluster_connected_triangles()
    cnt = np.asarray(cnt)
    bb = m.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound()))
    name = osp.basename(p)[:-4]
    rows.append(dict(mesh=name, components=int(len(cnt)),
                     faces=int(len(m.triangles)), bbox_diag=diag))
    print(f"{name:52s} comps={len(cnt):6d} faces={len(m.triangles):9,d} "
          f"bbox={diag:.3f}", flush=True)

json.dump(rows, open("banf_final_structure.json", "w"), indent=2)
