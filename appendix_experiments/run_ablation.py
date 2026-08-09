#!/usr/bin/env python
"""NeurIPS 2026 rebuttal — 2x2 ablation orchestrator.

Cells (factorial over residual structure x nested-band supervision):
  (a) residual + nested band       = full M-plicits (paper). Clean condition
      reuses the paper checkpoints in results/normalized_<shape>_gt.
  (b) residual, full-domain        = train_sdf_residual.py + train_sdf_residual_fine.py
  (c) non-residual, nested band    = train_sdf_on_neighborhood.py (residual: False), chained
  (d) single SIREN, matched budget = train_sdf.py, (350,3) ~= 247k params

Conditions: clean / noise (1% bbox-diagonal displacement along GT normals).
All metrics are computed against the CLEAN ground truth.

Run inside the `new_i3d` conda env:
  conda run -n new_i3d python run_ablation.py            # everything
  conda run -n new_i3d python run_ablation.py --smoke    # quick plumbing test
"""
import argparse
import csv
import os
import os.path as osp
import subprocess
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
I3D = osp.dirname(HERE)
SCRIPTS = osp.join(I3D, "experiment_scripts")
CFG = osp.join(HERE, "configs")
RESULTS = osp.join(HERE, "results")
MESHES = osp.join(HERE, "meshes")
METRICS = osp.join(HERE, "metrics")
# Set by --fresh-a: train cell (a)'s clean medium/fine here instead of reusing
# the paper checkpoints, so all cells share one training protocol.
FRESH_A = False

# Shape keys are the canonical basenames used across data/ and results/:
#   clean input:  data/normalized_sphere/input/<shape>.ply
#   noisy input:  data/normalized_noise/noise_data/<shape>.ply
#   paper ckpts:  results/<shape>/{coarse,medium,fine}/best.pth
STANFORD = [
    "normalized_armadillo_gt",
    "normalized_asian_dragon_gt",
    "normalized_lucy_gt",
    "normalized_thai_gt_half",
]
THINGI = ["44234", "64764", "68381", "72870", "73075", "77245"]
# Extra variants runnable on demand (not in the default queue):
# full-thai with the paper's real Feb-20 noise input.
EXTRA = ["normalized_thai_gt"]
SHAPES = STANFORD + THINGI
CONDS = ["clean", "noise"]
CELLS = ["a", "b", "c", "d"]


def input_ply(shape, cond):
    if cond == "clean":
        return osp.join(I3D, "data", "normalized_sphere", "input", f"{shape}.ply")
    return osp.join(I3D, "data", "normalized_noise", "noise_data", f"{shape}.ply")


def gt_clean_ply(shape):
    return input_ply(shape, "clean")


def paper_stage_dir(shape, stage):
    d = osp.join(I3D, "results", shape, stage)
    if osp.exists(osp.join(d, "best.pth")):
        return d
    return None


def run_logged(cmd, cwd, logpath):
    os.makedirs(osp.dirname(logpath), exist_ok=True)
    print(f"  RUN {' '.join(map(str, cmd))}")
    t0 = time.time()
    with open(logpath, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            list(map(str, cmd)), cwd=cwd, stdout=log, stderr=subprocess.STDOUT
        )
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = open(logpath, encoding="utf-8", errors="replace").readlines()[-25:]
        raise RuntimeError(
            f"Command failed ({proc.returncode}) after {dt:.0f}s: {cmd}\n" + "".join(tail)
        )
    print(f"  done in {dt:.0f}s (log: {osp.relpath(logpath, HERE)})")


def ensure_stage(outdir):
    return osp.exists(osp.join(outdir, "best.pth"))


def train_all(shape, cond, cells, smoke):
    py = sys.executable
    mesh = input_ply(shape, cond)
    base = osp.join(RESULTS, shape, cond)
    os.makedirs(base, exist_ok=True)
    ep = ["--nepochs", "60"] if smoke else []
    ep_single = ["--nepochs", "120"] if smoke else []

    # ---- coarse (shared by cells a, b, c) ----
    coarse_dir = osp.join(base, "coarse")
    if cond == "clean":
        paper = paper_stage_dir(shape, "coarse")
        if paper:
            coarse_dir = paper
            print(f"  coarse: reusing paper checkpoint {osp.relpath(paper, I3D)}")
    if not ensure_stage(coarse_dir):
        run_logged(
            [py, osp.join(I3D, "train_sdf.py"), mesh, coarse_dir,
             osp.join(CFG, "coarse.yaml")] + ep,
            I3D, osp.join(base, "logs", "coarse.log"),
        )
    coarse_best = osp.join(coarse_dir, "best.pth")

    stage_dirs = {"coarse": coarse_dir}

    # ---- cell (a): residual + nested band ----
    if "a" in cells:
        a_med = osp.join(base, "a_medium")
        a_fin = osp.join(base, "a_fine")
        if cond == "clean" and not FRESH_A:
            pm, pf = paper_stage_dir(shape, "medium"), paper_stage_dir(shape, "fine")
            if pm and pf:
                a_med, a_fin = pm, pf
                print(f"  cell a: reusing paper checkpoints {osp.relpath(pm, I3D)}")
        elif cond == "clean":
            print("  cell a: --fresh-a, training medium/fine under the "
                  "ablation protocol (paper checkpoints ignored)")
        if not ensure_stage(a_med):
            run_logged(
                [py, osp.join(SCRIPTS, "train_sdf_on_neighborhood.py"), mesh, a_med,
                 osp.join(CFG, "medium_band_res.yaml"), coarse_best] + ep,
                SCRIPTS, osp.join(base, "logs", "a_medium.log"),
            )
        if not ensure_stage(a_fin):
            run_logged(
                [py, osp.join(SCRIPTS, "train_sdf_on_neighborhood_fine.py"), mesh, a_fin,
                 osp.join(CFG, "fine_band_res.yaml"), coarse_best,
                 osp.join(a_med, "best.pth")] + ep,
                SCRIPTS, osp.join(base, "logs", "a_fine.log"),
            )
        stage_dirs["a_medium"], stage_dirs["a_fine"] = a_med, a_fin

    # ---- cell (b): residual, full-domain supervision ----
    if "b" in cells:
        b_med = osp.join(base, "b_medium")
        b_fin = osp.join(base, "b_fine")
        if not ensure_stage(b_med):
            run_logged(
                [py, osp.join(SCRIPTS, "train_sdf_residual.py"), mesh, b_med,
                 osp.join(CFG, "medium_full_res.yaml"), coarse_best] + ep,
                SCRIPTS, osp.join(base, "logs", "b_medium.log"),
            )
        if not ensure_stage(b_fin):
            run_logged(
                [py, osp.join(SCRIPTS, "train_sdf_residual_fine.py"), mesh, b_fin,
                 osp.join(CFG, "fine_full_res.yaml"), coarse_best,
                 osp.join(b_med, "best.pth")] + ep,
                SCRIPTS, osp.join(base, "logs", "b_fine.log"),
            )
        stage_dirs["b_medium"], stage_dirs["b_fine"] = b_med, b_fin

    # ---- cell (c): non-residual, nested band (chained standalone SDFs) ----
    if "c" in cells:
        c_l2 = osp.join(base, "c_level2")
        c_l3 = osp.join(base, "c_level3")
        if not ensure_stage(c_l2):
            run_logged(
                [py, osp.join(SCRIPTS, "train_sdf_on_neighborhood.py"), mesh, c_l2,
                 osp.join(CFG, "medium_band_nores.yaml"), coarse_best] + ep,
                SCRIPTS, osp.join(base, "logs", "c_level2.log"),
            )
        if not ensure_stage(c_l3):
            run_logged(
                [py, osp.join(SCRIPTS, "train_sdf_on_neighborhood.py"), mesh, c_l3,
                 osp.join(CFG, "fine_band_nores.yaml"),
                 osp.join(c_l2, "best.pth")] + ep,
                SCRIPTS, osp.join(base, "logs", "c_level3.log"),
            )
        stage_dirs["c_level2"], stage_dirs["c_level3"] = c_l2, c_l3

    # ---- cell (d): single SIREN, matched budget ----
    if "d" in cells:
        d_dir = osp.join(base, "d_single")
        if not ensure_stage(d_dir):
            run_logged(
                [py, osp.join(I3D, "train_sdf.py"), mesh, d_dir,
                 osp.join(CFG, "single_matched.yaml")] + ep_single,
                I3D, osp.join(base, "logs", "d_single.log"),
            )
        stage_dirs["d_single"] = d_dir

    return stage_dirs


def recon_and_eval(shape, cond, cells, stage_dirs, smoke, device="cuda:0"):
    """Reconstruct each cell's mesh (identical full-evaluation protocol) and
    store the AUXILIARY metrics (delta_1, nesting violation, training times).

    CD/Hausdorff/IoU are computed afterwards by run_metrics.py using the
    paper's own metric code (metrics conda env).
    """
    import torch
    from i3d.meshing import create_mesh, create_mesh_multistage
    from i3d.util import from_pth
    sys.path.insert(0, HERE)
    import evallib

    res = 128 if smoke else 512
    nest_res = 128 if smoke else 256
    os.makedirs(MESHES, exist_ok=True)
    os.makedirs(METRICS, exist_ok=True)

    def load(d):
        return from_pth(osp.join(d, "best.pth"), w0=1).eval().to(device)

    def train_time(d):
        p = osp.join(d, "metrics.csv")
        if not osp.exists(p):
            return None  # paper checkpoint or missing
        with open(p) as f:
            rows = list(csv.reader(f))
        return float(rows[1][0])

    mesh_input = input_ply(shape, cond)

    # Adaptive bands (Eq. 5, eps = 0.3) from the TRAINING inputs.
    coarse = load(stage_dirs["coarse"])
    import numpy as np
    import open3d as o3d
    v = np.asarray(o3d.io.read_triangle_mesh(mesh_input).vertices, dtype=np.float32)
    vt = torch.from_numpy(v).to(device)
    maxd = evallib.max_abs_at_points([coarse], vt, device=device)
    delta1 = 1.3 * maxd

    results = {"shape": shape, "cond": cond, "max_abs_f1": maxd, "delta1": delta1}

    def mesh_path_for(cell):
        # double-underscore separator: shape names contain single underscores
        return osp.join(MESHES, f"{shape}__{cond}__{cell}.ply")

    # Native inference per cell:
    #  (a) culled multiscale MC with adaptive deltas (the paper's inference);
    #  (b) full-sum MC (its residuals are trained domain-wide; culling would
    #      be UNSOUND for it - that is the nesting-violation point);
    #  (c) culled SELECT over chained standalone SDFs;
    #  (d) plain single-model MC.
    if "a" in cells:
        med, fin = load(stage_dirs["a_medium"]), load(stage_dirs["a_fine"])
        delta2 = 1.3 * evallib.max_abs_at_points([coarse, med], vt, device=device)
        # 2-voxel padding on the culling thresholds: standard MC hygiene so the
        # band boundary never cuts through surface-adjacent grid cells.
        pad = 2 * (2.0 / (res - 1))
        m = {"mesh": mesh_path_for("a"), "delta2": delta2, "mc_pad": pad}
        if not osp.exists(m["mesh"]):
            print(f"  recon {shape}/{cond}/a (N={res}, culled deltas="
                  f"[{delta1:.2e},{delta2:.2e}] + pad {pad:.2e})")
            *_, st = create_mesh_multistage(
                [coarse, med, fin], deltas=[delta1 + pad, delta2 + pad],
                filename=m["mesh"], N=res, max_batch=32**3, device=device)
            m["sampling_time_s"] = st
        m["nesting"] = evallib.nesting_violation(coarse, m["mesh"], delta1,
                                                 device=device)
        m["train_time_s"] = {k: train_time(stage_dirs[k])
                             for k in ("coarse", "a_medium", "a_fine")}
        results["a"] = m
    if "b" in cells:
        med, fin = load(stage_dirs["b_medium"]), load(stage_dirs["b_fine"])
        m = {"mesh": mesh_path_for("b")}
        if not osp.exists(m["mesh"]):
            print(f"  recon {shape}/{cond}/b (N={res}, full sum)")
            *_, st = create_mesh_multistage(
                [coarse, med, fin], deltas=[], filename=m["mesh"],
                N=res, max_batch=32**3, device=device)
            m["sampling_time_s"] = st
        m["nesting"] = evallib.nesting_violation(coarse, m["mesh"], delta1,
                                                 device=device)
        m["train_time_s"] = {k: train_time(stage_dirs[k])
                             for k in ("coarse", "b_medium", "b_fine")}
        results["b"] = m
    if "c" in cells:
        l2, l3 = load(stage_dirs["c_level2"]), load(stage_dirs["c_level3"])
        delta2c = 1.3 * evallib.max_abs_at_points([l2], vt, device=device)
        pad = 2 * (2.0 / (res - 1))
        m = {"mesh": mesh_path_for("c"), "delta2": delta2c, "mc_pad": pad}
        if not osp.exists(m["mesh"]):
            print(f"  recon {shape}/{cond}/c (N={res}, culled select)")
            st = evallib.mc_select([coarse, l2, l3],
                                   [delta1 + pad, delta2c + pad],
                                   m["mesh"], N=res, device=device)
            m["sampling_time_s"] = st
        m["train_time_s"] = {k: train_time(stage_dirs[k])
                             for k in ("coarse", "c_level2", "c_level3")}
        results["c"] = m
    if "d" in cells:
        m = {"mesh": mesh_path_for("d")}
        if not osp.exists(m["mesh"]):
            print(f"  recon {shape}/{cond}/d (N={res}, single)")
            t0 = time.time()
            create_mesh(load(stage_dirs["d_single"]), filename=m["mesh"],
                        N=res, max_batch=32**3, device=device)
            m["sampling_time_s"] = time.time() - t0
        m["train_time_s"] = {"d_single": train_time(stage_dirs["d_single"])}
        results["d"] = m

    # Merge into any existing record rather than overwriting it: a partial
    # re-run (e.g. --cells a) must not drop the other cells' train_time_s,
    # delta2 and nesting entries, which summarize.py joins against.
    out = osp.join(METRICS, f"{shape}__{cond}.json")
    if osp.exists(out):
        import json
        with open(out, encoding="utf-8") as f:
            merged = json.load(f)
        merged.update(results)
        results = merged
    evallib.save_json(out, results)
    print(f"  aux metrics -> {osp.relpath(out, HERE)}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=SHAPES,
                    choices=SHAPES + EXTRA)
    ap.add_argument("--conds", nargs="+", default=CONDS, choices=CONDS)
    ap.add_argument("--cells", nargs="+", default=CELLS, choices=CELLS)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny epochs + low-res recon, for plumbing tests")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, only reconstruct + evaluate")
    ap.add_argument("--fresh-a", action="store_true",
                    help="train cell (a)'s medium/fine under the ablation "
                         "protocol on the clean condition instead of reusing "
                         "the paper checkpoints. Without this, clean (a) is "
                         "the paper's March run while (b)/(c) are trained "
                         "here - the cells are then not protocol-matched.")
    args = ap.parse_args()
    global FRESH_A
    FRESH_A = args.fresh_a

    if args.smoke:
        # Isolate smoke artifacts so they never shadow real checkpoints.
        global RESULTS, MESHES, METRICS
        RESULTS = osp.join(HERE, "smoke", "results")
        MESHES = osp.join(HERE, "smoke", "meshes")
        METRICS = osp.join(HERE, "smoke", "metrics")

    t0 = time.time()
    failures = []
    for shape in args.shapes:
        for cond in args.conds:
            print(f"=== {shape} / {cond} ===")
            if not osp.exists(input_ply(shape, cond)):
                print(f"  [skip] input missing: {input_ply(shape, cond)}")
                continue
            try:
                stage_dirs = train_all(shape, cond, args.cells, args.smoke)
                recon_and_eval(shape, cond, args.cells, stage_dirs, args.smoke)
            except Exception as e:
                print(f"  [FAILED] {shape}/{cond}: {e}")
                failures.append((shape, cond, str(e)[:300]))
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min; {len(failures)} failures")
    for shape, cond, err in failures:
        print(f"  FAILED {shape}/{cond}: {err}")
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
