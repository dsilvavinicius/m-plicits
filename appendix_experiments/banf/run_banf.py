"""Run the BANF re-implementation over the rebuttal shape set.

Produces meshes/<shape>__<cond>__banf.ply (full recomposition, 512^3 MC) and
meshes/<shape>__<cond>__banfcoarse.ply (level-0 only, the 32^3-band field),
metered later by run_metrics.py with the paper protocol.

Usage:
  python run_banf.py                     # all 10 shapes x clean,noise
  python run_banf.py --shapes A B --conds clean --smoke
"""
import argparse
import os
import os.path as osp
import sys
import time

import torch

HERE = osp.dirname(osp.abspath(__file__))
REB = osp.dirname(HERE)
I3D = osp.dirname(REB)
sys.path.insert(0, HERE)
from banf_sdf import train_banf, extract_mesh  # noqa: E402

STANFORD = ["normalized_armadillo_gt", "normalized_asian_dragon_gt",
            "normalized_lucy_gt", "normalized_thai_gt_half"]
THINGI = ["44234", "64764", "68381", "72870", "73075", "77245"]


def input_ply(shape, cond):
    if cond == "clean":
        return osp.join(I3D, "data", "normalized_sphere", "input", f"{shape}.ply")
    return osp.join(I3D, "data", "normalized_noise", "noise_data", f"{shape}.ply")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=STANFORD + THINGI)
    ap.add_argument("--conds", nargs="+", default=["clean", "noise"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    meshes = osp.join(REB, "meshes")
    os.makedirs(meshes, exist_ok=True)
    kw = {}
    res_n = 512
    if args.smoke:
        kw = {"iters_first": 200, "iters_rest": 200, "batch": 20_000,
              "log_every": 100}
        res_n = 128

    for shape in args.shapes:
        for cond in args.conds:
            src = input_ply(shape, cond)
            if not osp.exists(src):
                print(f"[skip] {src}")
                continue
            out_full = osp.join(meshes, f"{shape}__{cond}__banf.ply")
            out_coarse = osp.join(meshes, f"{shape}__{cond}__banfcoarse.ply")
            if osp.exists(out_full) and osp.exists(out_coarse):
                print(f"[cached] {shape}/{cond}")
                continue
            print(f"=== BANF {shape}/{cond} ===", flush=True)
            t0 = time.time()
            cascade = train_banf(src, **kw)
            t_train = time.time() - t0
            extract_mesh(cascade, out_full, N=res_n)
            extract_mesh(cascade, out_coarse, N=res_n, upto=1)
            print(f"{shape}/{cond}: trained {t_train:.0f}s -> {out_full}", flush=True)
            with open(osp.join(HERE, "banf_times.csv"), "a", encoding="utf-8") as f:
                f.write(f"{shape},{cond},{t_train:.1f}\n")
            del cascade
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
