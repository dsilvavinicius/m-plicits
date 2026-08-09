"""Culled vs unculled extraction for ablation cells (a) and (c).

Motivation: the main ablation extracts BOTH (a) and (c) with band-culled
inference (coarse field outside the band, finer field only inside). That is
symmetric and fair, but it hides what suppl. Fig. S10 actually claims —
that band-training WITHOUT residuals leaves spurious components outside the
band, and the residual sum removes them. Culling never evaluates the finer
field out there, so the defect is invisible.

This script re-extracts the same trained models with NO culling:
  (a) uncull: the full sum f1 + r1 + r2 evaluated over the whole domain
  (c) uncull: the standalone finest SDF f3 evaluated over the whole domain

Prediction if the paper's claim holds: (a) barely changes, (c) degrades.

No retraining. Outputs meshes/<shape>__<cond>__{aNC,cNC}.ply, metered
afterwards by run_metrics.py with the paper's protocol.

Usage: conda run -n new_i3d python run_nocull.py [--shapes ...] [--conds ...]
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import open3d as o3d
import torch

HERE = osp.dirname(osp.abspath(__file__))
I3D = osp.dirname(HERE)
sys.path.insert(0, I3D)
sys.path.insert(0, HERE)
from i3d.meshing import create_mesh, create_mesh_multistage  # noqa: E402
from i3d.util import from_pth  # noqa: E402

STANFORD = ["normalized_armadillo_gt", "normalized_asian_dragon_gt",
            "normalized_lucy_gt", "normalized_thai_gt_half"]
THINGI = ["44234", "64764", "68381", "72870", "73075", "77245"]
DEV = "cuda:0"


def stage(shape, cond, name):
    return osp.join(HERE, "results", shape, cond, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=STANFORD + THINGI)
    ap.add_argument("--conds", nargs="+", default=["clean", "noise"])
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()

    out = osp.join(HERE, "meshes_nocull")
    os.makedirs(out, exist_ok=True)
    log = {}
    for shape in args.shapes:
        for cond in args.conds:
            # On the clean condition cell (a) reuses the paper checkpoints
            # (results/<shape>/{coarse,medium,fine}); on noise it trains its
            # own. Fall back to the paper tree for any stage not found here.
            def pick(local, paper):
                return local if osp.exists(osp.join(local, "best.pth")) else paper

            cdir = pick(stage(shape, cond, "coarse"),
                        osp.join(I3D, "results", shape, "coarse"))
            amed = pick(stage(shape, cond, "a_medium"),
                        osp.join(I3D, "results", shape, "medium"))
            afin = pick(stage(shape, cond, "a_fine"),
                        osp.join(I3D, "results", shape, "fine"))
            need = [cdir, amed, afin, stage(shape, cond, "c_level3")]
            if not all(osp.exists(osp.join(d, "best.pth")) for d in need):
                print(f"[skip] {shape}/{cond}: checkpoints not local")
                continue
            print(f"=== {shape}/{cond} ===", flush=True)
            coarse = from_pth(osp.join(cdir, "best.pth"), w0=1).eval().to(DEV)
            a_med = from_pth(osp.join(need[1], "best.pth"), w0=1).eval().to(DEV)
            a_fin = from_pth(osp.join(need[2], "best.pth"), w0=1).eval().to(DEV)
            c_l3 = from_pth(osp.join(need[3], "best.pth"), w0=1).eval().to(DEV)

            # (a) unculled: full residual sum everywhere
            pa = osp.join(out, f"{shape}__{cond}__aNC.ply")
            if not osp.exists(pa):
                create_mesh_multistage([coarse, a_med, a_fin], deltas=[],
                                       filename=pa, N=args.res,
                                       max_batch=32**3, device=DEV)
            # (c) unculled: standalone finest SDF everywhere
            pc = osp.join(out, f"{shape}__{cond}__cNC.ply")
            if not osp.exists(pc):
                create_mesh(c_l3, filename=pc, N=args.res,
                            max_batch=32**3, device=DEV)

            # structural signal: connected components / bbox
            rec = {}
            for tag, p in (("aNC", pa), ("cNC", pc)):
                m = o3d.io.read_triangle_mesh(p)
                _, cnt, _ = m.cluster_connected_triangles()
                cnt = np.asarray(cnt)
                bb = m.get_axis_aligned_bounding_box()
                rec[tag] = {
                    "components": int(len(cnt)),
                    "bbox_diag": float(np.linalg.norm(bb.get_max_bound()
                                                      - bb.get_min_bound())),
                    "verts": int(len(m.vertices)),
                }
                print(f"  {tag}: comps={rec[tag]['components']} "
                      f"bbox={rec[tag]['bbox_diag']:.3f}", flush=True)
            log[f"{shape}__{cond}"] = rec
            with open(osp.join(HERE, "nocull_structure.json"), "w") as f:
                json.dump(log, f, indent=2)
            del coarse, a_med, a_fin, c_l3
            torch.cuda.empty_cache()
    print("done -> run_metrics.py --meshes-dir meshes_nocull "
          "--out-prefix nocull_metrics")


if __name__ == "__main__":
    main()
