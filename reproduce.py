#!/usr/bin/env python
"""Reproduce the paper's tables from the released checkpoints, end to end.

Run from the repository root with the data archive unzipped in place and the
training environment active. Each stage writes under reproduction/ and the
final report compares every number with the published one.

    python reproduce.py recon            # meshes at 512^3 for every released shape (Tab. 2 sampling, Tab. 5)
    python reproduce.py metrics          # CD / Hausdorff / IoU with the paper's metric code (Tab. 2)
    python reproduce.py fps              # real-time renderer rows on the Armadillo (Tab. 4; Windows build)
    python reproduce.py train --shapes normalized_armadillo_gt     # retrain the three stages from scratch
    python reproduce.py noise --shapes normalized_armadillo_gt     # retrain on the 1% noisy input (Tab. 3)
    python reproduce.py report           # reproduction/REPORT.md
    python reproduce.py all              # recon + metrics + fps + report (no retraining)

The metric code needs PyTorch3D, i.e. the `metrics` conda environment
(metrics/environment.yml). Point MPLICITS_METRICS_PYTHON at its interpreter
if it is not found automatically.
"""
import argparse
import csv
import glob
import json
import os
import os.path as osp
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = osp.dirname(osp.abspath(__file__))
OUT = osp.join(ROOT, "reproduction")
GT_DIR = osp.join(ROOT, "data", "normalized_sphere", "input")
NOISE_DIR = osp.join(ROOT, "data", "normalized_noise", "noise_data")
RESULTS = osp.join(ROOT, "results")
STAGES = ("coarse", "medium", "fine")
RESOLUTION = 512
NUM_POINTS = 500_000

# Published values (NeurIPS 2026 submission). Tab. 2 / Tab. 3 aggregate over
# the 37 / 36 shapes; Tab. 4 is the Armadillo on an RTX 5090.
PAPER = {
    "tab2": {
        "coarse": dict(mean_cd=5.36e-5, median_cd=3.80e-5, mean_iou=0.448, median_iou=0.459,
                       params=17153, train_min=4.07, sampling_s=1.51, fps=180),
        "fine": dict(mean_cd=6.57e-5, median_cd=1.87e-5, mean_iou=0.586, median_iou=0.867,
                     params=246627, train_min=8.53, sampling_s=5.29, fps=43),
    },
    "tab3": {
        "coarse": dict(mean_cd=7.42e-5, median_cd=5.33e-5, mean_iou=0.381, median_iou=0.403),
        "medium": dict(mean_cd=4.62e-5, median_cd=3.03e-5, mean_iou=0.537, median_iou=0.638),
        "fine": dict(mean_cd=3.36e-5, median_cd=1.62e-5, mean_iou=0.582, median_iou=0.591),
    },
    "tab4": [  # (row label, experiment, flags, paper FPS)
        ("(400,4) SIREN baseline", "armadillo_siren", "-iters=20,0,0", 19),
        ("(128,2) coarse", "armadillo", "-iters=20,0,0", 180),
        ("(128,2)>(256,2) (NM)", "armadillo", "-iters=20,0,0 -delta=0.02 -normal_lod=1", 85),
        ("(128,2)>(256,2)", "armadillo", "-iters=20,5,0 -delta=0.02 -normal_lod=1", 70),
        ("(128,2)>(256,2)>(400,2) (NM)", "armadillo", "-iters=20,5,0 -delta=0.02 -normal_lod=2", 43),
        ("(128,2)>(256,2)>(400,2)", "armadillo", "-iters=20,5,5 -delta=0.02 -normal_lod=2", 35),
    ],
}


# ----------------------------------------------------------------------------- helpers
def log(msg):
    print(f"[reproduce] {msg}", flush=True)


def run(cmd, cwd=ROOT, capture=False):
    log(" ".join(cmd))
    if capture:
        return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout
    subprocess.run(cmd, cwd=cwd, check=True)


def released_shapes():
    """Shapes with coarse+medium+fine checkpoints. A trailing .ply in a
    results folder name is tolerated (older archives)."""
    shapes = {}
    for d in sorted(glob.glob(osp.join(RESULTS, "*"))):
        if not osp.isdir(d):
            continue
        name = osp.basename(d)
        stem = name[:-4] if name.endswith(".ply") else name
        ck = {s: osp.join(d, s, "best.pth") for s in STAGES}
        if not all(osp.exists(p) for p in ck.values()):
            alt = osp.join(RESULTS, stem + ".ply")
            for s in STAGES:
                if not osp.exists(ck[s]) and osp.exists(osp.join(alt, s, "best.pth")):
                    ck[s] = osp.join(alt, s, "best.pth")
        if all(osp.exists(p) for p in ck.values()) and osp.exists(osp.join(GT_DIR, stem + ".ply")):
            shapes[stem] = ck
    return shapes


def select(shapes, wanted):
    if not wanted:
        return shapes
    missing = [w for w in wanted if w not in shapes]
    if missing:
        sys.exit(f"unknown shapes {missing}; available: {sorted(shapes)}")
    return {w: shapes[w] for w in wanted}


def save_json(path, obj):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path, default=None):
    if osp.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def metrics_python():
    """Command prefix that runs Python inside the `metrics` environment.

    `conda run` is preferred: it activates the environment, which on Windows
    is what puts open3d's DLL dependencies on the PATH (calling the
    interpreter directly fails with "DLL load failed while importing pybind").
    """
    env = os.environ.get("MPLICITS_METRICS_PYTHON")
    if env:
        return [env]
    root = osp.dirname(osp.dirname(sys.prefix))          # <conda root>/envs/<this env>
    for conda in (os.environ.get("CONDA_EXE"), shutil.which("conda"),
                  osp.join(root, "Scripts", "conda.exe"), osp.join(root, "bin", "conda"),
                  osp.join(root, "condabin", "conda.bat")):
        if conda and osp.exists(conda):
            return [conda, "run", "-n", "metrics", "--no-capture-output", "python"]
    envs = osp.dirname(sys.prefix)
    for cand in (osp.join(envs, "metrics", "python.exe"), osp.join(envs, "metrics", "bin", "python")):
        if osp.exists(cand):
            return [cand]
    sys.exit("metrics environment not found: create metrics/environment.yml or set MPLICITS_METRICS_PYTHON")


# ----------------------------------------------------------------------------- stages
def stage_recon(shapes, cond="clean", ckpts=None):
    """Coarse-only and multiscale (culled) meshes, plus an unculled multiscale
    extraction for the Tab. 5 analogue. Times come from reconstruct.py."""
    ckpts = ckpts or shapes
    mesh_dir = osp.join(OUT, "meshes", cond)
    os.makedirs(mesh_dir, exist_ok=True)
    times_path = osp.join(OUT, f"recon_times_{cond}.json")
    times = load_json(times_path, {})
    for shape, ck in ckpts.items():
        jobs = {
            "coarse": [ck["coarse"]],
            "fine": [ck["fine"], "--multistage", "--coarse_path", ck["coarse"], "--medium_path", ck["medium"]],
            "fine_unculled": [ck["fine"], "--coarse_path", ck["coarse"], "--medium_path", ck["medium"]],
        }
        if cond != "clean":
            jobs["medium"] = [ck["medium"], "--multistage", "--coarse_path", ck["coarse"]]
        for level, extra in jobs.items():
            out = osp.join(mesh_dir, f"{shape}__{level}.ply")
            if osp.exists(out) and level in times.get(shape, {}):
                continue
            t0 = time.time()
            txt = run([sys.executable, "reconstruct.py", extra[0], out, "-r", str(RESOLUTION),
                       "--device", "cuda"] + extra[1:], capture=True)
            m = re.search(r"SAMPLING_TIME_S:([0-9.eE+-]+)", txt)
            times.setdefault(shape, {})[level] = dict(sampling_s=float(m.group(1)) if m else None,
                                                     wall_s=time.time() - t0)
            save_json(times_path, times)
    return times


def stage_metrics(cond="clean"):
    mesh_dir = osp.join(OUT, "meshes", cond)
    pairs = []
    for m in sorted(glob.glob(osp.join(mesh_dir, "*__*.ply"))):
        shape = osp.basename(m).split("__")[0]
        gt = osp.join(GT_DIR, shape + ".ply")
        if osp.exists(gt):
            pairs.append({"mesh1": m, "mesh2": gt})
    if not pairs:
        sys.exit(f"no meshes in {mesh_dir}; run recon first")
    pairs_json = osp.join(OUT, f"pairs_{cond}.json")
    save_json(pairs_json, pairs)
    csv_out = osp.join(OUT, f"metrics_{cond}.csv")
    run(metrics_python() + [osp.join(ROOT, "metrics", "meshes", "compute_distance_metrics.py"),
                            "--json_file", pairs_json, "--csv_file", csv_out,
                            "--latex_file", osp.join(OUT, f"metrics_{cond}.tex"),
                            "--num_points", str(NUM_POINTS), "--norm", "2", "--iou_voxel_size", "0.01"])
    return csv_out


def stage_paper_noise():
    """Tab. 3 without retraining: metrics of the paper's own noise
    reconstructions (optional archive m-plicits-paper-meshes.zip, unpacked as
    paper_meshes/noise/<shape>__<level>.ply) against the clean ground truth."""
    src = osp.join(ROOT, "paper_meshes", "noise")
    if not osp.isdir(src):
        sys.exit("paper_meshes/noise not found: python tools/download_data.py --paper-meshes")
    dst = osp.join(OUT, "meshes", "paper_noise")
    os.makedirs(dst, exist_ok=True)
    for m in glob.glob(osp.join(src, "*__*.ply")):
        target = osp.join(dst, osp.basename(m))
        if not osp.exists(target):
            os.symlink(m, target) if hasattr(os, "symlink") and os.name != "nt" else shutil.copy2(m, target)
    return stage_metrics(cond="paper_noise")


def stage_fps():
    exe = osp.join(ROOT, "renderer", "cuda", "build", "Release", "MIP-plicitsRenderer.exe")
    if not osp.exists(exe):
        log("renderer not built (see renderer/README.md); skipping fps")
        return None
    rows = []
    for label, exp, flags, paper in PAPER["tab4"]:
        txt = subprocess.run([exe, f"-experiment={exp}", "-benchmark=500"] + flags.split(),
                             cwd=osp.dirname(exe), capture_output=True, text=True).stdout
        m = re.search(r"avg_fps=([0-9.]+)", txt)
        fps = float(m.group(1)) if m else None
        rows.append(dict(row=label, experiment=exp, flags=flags, fps=fps, paper_fps=paper))
        log(f"{label}: {fps} FPS (paper {paper})")
    save_json(osp.join(OUT, "fps.json"), rows)
    return rows


def train_chain(shape, ck, input_ply, out_root):
    """Three stages with the configs shipped next to the released checkpoints."""
    cfg = {s: osp.join(osp.dirname(ck[s]), "config.yaml") for s in STAGES}
    out = {s: osp.join(out_root, shape, s) for s in STAGES}
    timings = {}
    t0 = time.time()
    run([sys.executable, "train_sdf.py", input_ply, out["coarse"], cfg["coarse"]])
    timings["coarse"] = time.time() - t0
    t0 = time.time()
    run([sys.executable, "experiment_scripts/train_sdf_on_neighborhood.py", input_ply, out["medium"],
         cfg["medium"], osp.join(out["coarse"], "best.pth")])
    timings["medium"] = time.time() - t0
    t0 = time.time()
    run([sys.executable, "experiment_scripts/train_sdf_on_neighborhood_fine.py", input_ply, out["fine"],
         cfg["fine"], osp.join(out["coarse"], "best.pth"), osp.join(out["medium"], "best.pth")])
    timings["fine"] = time.time() - t0
    return {s: osp.join(out[s], "best.pth") for s in STAGES}, timings


def stage_train(shapes, cond):
    """cond=clean: retrain from the clean input. cond=noise: from the 1% noisy
    input, evaluated against the clean ground truth (Tab. 3)."""
    out_root = osp.join(OUT, "train_" + cond)
    times_path = osp.join(OUT, f"train_times_{cond}.json")
    times = load_json(times_path, {})
    new_ckpts = {}
    for shape, ck in shapes.items():
        inp = osp.join(GT_DIR if cond == "clean" else NOISE_DIR, shape + ".ply")
        if not osp.exists(inp):
            log(f"no {cond} input for {shape}; skipping")
            continue
        new_ck, t = train_chain(shape, ck, inp, out_root)
        times[shape] = t
        save_json(times_path, times)
        new_ckpts[shape] = new_ck
    if new_ckpts:
        mesh_cond = "retrained" if cond == "clean" else "noise"
        stage_recon(shapes, cond=mesh_cond, ckpts=new_ckpts)
        stage_metrics(cond=mesh_cond)


# ----------------------------------------------------------------------------- report
def read_metrics(cond):
    p = osp.join(OUT, f"metrics_{cond}.csv")
    if not osp.exists(p):
        return {}
    per = {}
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            shape, level = row["File1"][:-4].split("__")
            per.setdefault(level, {})[shape] = dict(cd=float(row["Chamfer"]), hd=float(row["Hausdorff"]),
                                                    iou=float(row["IoU"]))
    return per


def agg(d):
    cds = [v["cd"] for v in d.values()]
    ious = [v["iou"] for v in d.values()]
    return dict(n=len(cds), mean_cd=statistics.fmean(cds), median_cd=statistics.median(cds),
                mean_iou=statistics.fmean(ious), median_iou=statistics.median(ious))


def fmt(x, sci=False):
    if x is None:
        return "n/a"
    return f"{x:.2E}" if sci else f"{x:.3f}"


def stage_report():
    lines = ["# Reproduction report", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} on {platform.node()} ({platform.platform()}).", ""]
    clean = read_metrics("clean")
    if clean:
        lines += ["## Tab. 2, ours rows (released checkpoints, clean inputs)", "",
                  "| level | shapes | mean CD | median CD | mean IoU | median IoU |", "|---|---|---|---|---|---|"]
        for level in ("coarse", "fine"):
            if level in clean:
                a = agg(clean[level])
                p = PAPER["tab2"][level]
                lines.append(f"| {level} (reproduced) | {a['n']} | {fmt(a['mean_cd'], 1)} | {fmt(a['median_cd'], 1)} "
                             f"| {fmt(a['mean_iou'])} | {fmt(a['median_iou'])} |")
                lines.append(f"| {level} (paper) | 37 | {fmt(p['mean_cd'], 1)} | {fmt(p['median_cd'], 1)} "
                             f"| {fmt(p['mean_iou'])} | {fmt(p['median_iou'])} |")
        if "fine_unculled" in clean:
            a = agg(clean["fine_unculled"])
            lines.append(f"| fine, unculled extraction | {a['n']} | {fmt(a['mean_cd'], 1)} | {fmt(a['median_cd'], 1)} "
                         f"| {fmt(a['mean_iou'])} | {fmt(a['median_iou'])} |")
        lines.append("")
    times = load_json(osp.join(OUT, "recon_times_clean.json"), {})
    if times:
        def mean_t(level):
            v = [t[level]["sampling_s"] for t in times.values() if level in t and t[level]["sampling_s"] is not None]
            return statistics.fmean(v) if v else None
        lines += ["## Extraction time at 512^3 (Tab. 2 sampling column; Tab. 5 analogue)", "",
                  "| extraction | shapes | mean seconds | paper |", "|---|---|---|---|",
                  f"| coarse only | {len(times)} | {fmt(mean_t('coarse'))} | {PAPER['tab2']['coarse']['sampling_s']} |",
                  f"| multiscale, band-culled | {len(times)} | {fmt(mean_t('fine'))} | {PAPER['tab2']['fine']['sampling_s']} |",
                  f"| multiscale, unculled (all levels everywhere) | {len(times)} | {fmt(mean_t('fine_unculled'))} "
                  f"| Tab. 5 baseline analogue |", ""]
    fps = load_json(osp.join(OUT, "fps.json"))
    if fps:
        lines += ["## Tab. 4, real-time renderer (Armadillo, 512^2, 500 frames)", "",
                  "| row | reproduced FPS | paper FPS | flags |", "|---|---|---|---|"]
        for r in fps:
            lines.append(f"| {r['row']} | {fmt(r['fps'])} | {r['paper_fps']} | `-experiment={r['experiment']} {r['flags']}` |")
        lines.append("")
    pn = read_metrics("paper_noise")
    if pn:
        lines += ["## Tab. 3, ours rows (the paper's noise reconstructions, clean ground truth)", "",
                  "| level | shapes | mean CD | median CD | mean IoU | median IoU |", "|---|---|---|---|---|---|"]
        for level in ("coarse", "medium", "fine"):
            if level in pn:
                a = agg(pn[level])
                p = PAPER["tab3"][level]
                lines.append(f"| {level} (reproduced) | {a['n']} | {fmt(a['mean_cd'], 1)} | {fmt(a['median_cd'], 1)} "
                             f"| {fmt(a['mean_iou'])} | {fmt(a['median_iou'])} |")
                lines.append(f"| {level} (paper) | 36 | {fmt(p['mean_cd'], 1)} | {fmt(p['median_cd'], 1)} "
                             f"| {fmt(p['mean_iou'])} | {fmt(p['median_iou'])} |")
        lines.append("")
    for cond, title in (("retrained", "Retrained from scratch (clean input) vs. released"),
                        ("noise", "Tab. 3, retrained on the 1% noisy input, evaluated against the clean ground truth")):
        m = read_metrics(cond)
        if not m:
            continue
        lines += [f"## {title}", "", "| shape | level | CD | IoU |", "|---|---|---|---|"]
        for level in ("coarse", "medium", "fine"):
            for shape, v in sorted(m.get(level, {}).items()):
                ref = ""
                if cond == "retrained" and level in clean and shape in clean[level]:
                    ref = f" (released: {fmt(clean[level][shape]['cd'], 1)} / {fmt(clean[level][shape]['iou'])})"
                lines.append(f"| {shape} | {level} | {fmt(v['cd'], 1)} | {fmt(v['iou'])}{ref} |")
        if cond == "noise":
            for level in ("coarse", "medium", "fine"):
                if level in m:
                    a = agg(m[level])
                    p = PAPER["tab3"][level]
                    lines.append(f"| subset mean ({a['n']}) | {level} | {fmt(a['mean_cd'], 1)} | {fmt(a['mean_iou'])} "
                                 f"(paper, 36 shapes: {fmt(p['mean_cd'], 1)} / {fmt(p['mean_iou'])}) |")
        tt = load_json(osp.join(OUT, f"train_times_{'clean' if cond == 'retrained' else 'noise'}.json"), {})
        if tt:
            lines.append("")
            for shape, t in tt.items():
                lines.append(f"Training time {shape}: coarse {t['coarse'] / 60:.1f} min, medium {t['medium'] / 60:.1f} min, "
                             f"fine {t['fine'] / 60:.1f} min (paper averages: coarse {PAPER['tab2']['coarse']['train_min']}, "
                             f"full {PAPER['tab2']['fine']['train_min']}).")
        lines.append("")
    os.makedirs(OUT, exist_ok=True)
    with open(osp.join(OUT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["recon", "metrics", "fps", "train", "noise", "paper-noise", "report", "all"])
    ap.add_argument("--shapes", nargs="*", default=None, help="subset of released shapes (default: all)")
    args = ap.parse_args()
    shapes = select(released_shapes(), args.shapes)
    log(f"{len(shapes)} released shapes")
    if args.stage in ("recon", "all"):
        stage_recon(shapes)
    if args.stage in ("metrics", "all"):
        stage_metrics()
    if args.stage in ("fps", "all"):
        stage_fps()
    if args.stage == "train":
        stage_train(shapes, "clean")
    if args.stage == "noise":
        stage_train(shapes, "noise")
    if args.stage == "paper-noise":
        stage_paper_noise()
    if args.stage in ("report", "all"):
        stage_report()


if __name__ == "__main__":
    main()
