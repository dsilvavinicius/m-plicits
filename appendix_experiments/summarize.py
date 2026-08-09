#!/usr/bin/env python
"""Join paper-protocol metrics (paper_metrics.csv) with the auxiliary metrics
(metrics/*.json: delta_1, training time) into the final rebuttal tables.

Aggregates over the canonical 10-shape set (4 Stanford + 6 Thingi32).
The full-thai noise variant (normalized_thai_gt) is reported separately.
"""
import csv
import glob
import json
import os.path as osp
from collections import defaultdict

HERE = osp.dirname(osp.abspath(__file__))

CELL_LABELS = {
    "a": "(a) residual + nested band (M-plicits)",
    "b": "(b) residual, full-domain",
    "c": "(c) non-residual, nested band",
    "d": "(d) single SIREN, matched params",
    "spsr": "Screened Poisson (depth 10, trimmed)",
    "spsrnotrim": "Screened Poisson (depth 10, NO trim)",
    "banf": "BANF re-impl (full recomposition)",
    "banfcoarse": "BANF re-impl (level 0 only)",
}
CELL_ORDER = ["a", "b", "c", "d", "spsr", "spsrnotrim", "banf", "banfcoarse"]

STANFORD = ["normalized_armadillo_gt", "normalized_asian_dragon_gt",
            "normalized_lucy_gt", "normalized_thai_gt_half"]
THINGI = ["44234", "64764", "68381", "72870", "73075", "77245"]
SHAPES_10 = STANFORD + THINGI
FAIL_IOU = 0.35          # "catastrophic reconstruction" threshold


def total_time(tt):
    if not tt:
        return None
    vals = [v for v in tt.values() if v is not None]
    return sum(vals) if vals else None


def load_aux():
    aux = {}
    for p in glob.glob(osp.join(HERE, "metrics", "*__*.json")):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        aux[(d["shape"], d["cond"])] = d
    return aux


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    csv_path = osp.join(HERE, "paper_metrics.csv")
    if not osp.exists(csv_path):
        print("paper_metrics.csv missing - run run_metrics.py first")
        return
    aux = load_aux()

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            name = rec["File1"]
            base = name[:-4] if name.endswith(".ply") else name
            parts = base.split("__")
            if len(parts) != 3:
                continue
            shape, cond, cell = parts
            a = aux.get((shape, cond), {})
            cell_aux = a.get(cell, {})
            iou = rec["IoU"]
            rows.append({
                "shape": shape, "cond": cond, "cell": cell,
                "cd": float(rec["Chamfer"]),
                "hausdorff": float(rec["Hausdorff"]),
                "iou": float(iou) if iou not in ("", "nan") else None,
                "train_time_s": total_time(cell_aux.get("train_time_s")),
                "delta1": a.get("delta1"),
            })
    if not rows:
        print("No parsed rows.")
        return

    with open(osp.join(HERE, "ablation_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg_block(rws, title):
        by = defaultdict(list)
        for r in rws:
            by[(r["cell"], r["cond"])].append(r)
        out = [f"## {title}", ""]
        for cond in ("clean", "noise"):
            present = [c for c in CELL_ORDER if (c, cond) in by]
            if not present:
                continue
            out += [f"### {cond}", "",
                    "| Variant | mean CD | median CD | mean Hausdorff | mean IoU | "
                    "median IoU | failures (IoU<0.35) | n |",
                    "|---|---|---|---|---|---|---|---|"]
            for cell in present:
                rs = by[(cell, cond)]
                cds = [x["cd"] for x in rs]
                hs = [x["hausdorff"] for x in rs]
                ious = [x["iou"] for x in rs if x["iou"] is not None]
                fails = sum(1 for x in ious if x < FAIL_IOU)
                out.append(
                    f"| {CELL_LABELS.get(cell, cell)} | {sum(cds)/len(cds):.3e} | "
                    f"{median(cds):.3e} | {sum(hs)/len(hs):.3e} | "
                    f"{(sum(ious)/len(ious) if ious else float('nan')):.3f} | "
                    f"{(median(ious) if ious else float('nan')):.3f} | "
                    f"{fails}/{len(ious)} | {len(rs)} |")
            out.append("")
        return out

    main_rows = [r for r in rows if r["shape"] in SHAPES_10]
    extra_rows = [r for r in rows if r["shape"] not in SHAPES_10]

    lines = ["# Rebuttal ablation — paper-protocol metrics",
             "",
             "Protocol: 500K samples, L2 Chamfer, Open3D voxel IoU (0.01); "
             "reconstruction at 512^3. Identical architectures, omega_0 "
             "schedule, epochs and seed across cells (a)-(d).", ""]
    lines += agg_block(main_rows, "All 10 shapes (4 Stanford + 6 Thingi32)")
    lines += agg_block([r for r in main_rows if r["shape"] in STANFORD],
                       "Stanford only")
    lines += agg_block([r for r in main_rows if r["shape"] in THINGI],
                       "Thingi32 subset only")
    if extra_rows:
        lines += agg_block(extra_rows, "Variants outside the 10-shape set "
                                       "(full-thai noise)")

    # Training-time comparison (a) vs (d)
    lines += ["## Training time, cells (a) vs (d)", "",
              "| Shape | Cond | (a) staged total (s) | (d) monolith (s) | ratio |",
              "|---|---|---|---|---|"]
    tt = {(r["shape"], r["cond"], r["cell"]): r["train_time_s"] for r in rows}
    ra, rd = [], []
    for s in SHAPES_10:
        for c in ("clean", "noise"):
            a_t, d_t = tt.get((s, c, "a")), tt.get((s, c, "d"))
            if a_t and d_t:
                ra.append(a_t)
                rd.append(d_t)
                lines.append(f"| {s} | {c} | {a_t:.0f} | {d_t:.0f} | {d_t/a_t:.2f}x |")
    if ra:
        lines.append(f"| **mean** | | **{sum(ra)/len(ra):.0f}** | "
                     f"**{sum(rd)/len(rd):.0f}** | "
                     f"**{(sum(rd)/len(rd))/(sum(ra)/len(ra)):.2f}x** |")
    lines.append("")

    lines += ["## Per-shape rows", "",
              "| Shape | Cond | Variant | CD | Hausdorff | IoU | Train (s) |",
              "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["shape"], x["cond"],
                                         CELL_ORDER.index(x["cell"])
                                         if x["cell"] in CELL_ORDER else 99)):
        iou = f"{r['iou']:.3f}" if r["iou"] is not None else "-"
        tt_s = f"{r['train_time_s']:.0f}" if r["train_time_s"] else "-"
        lines.append(f"| {r['shape']} | {r['cond']} | {r['cell']} | "
                     f"{r['cd']:.3e} | {r['hausdorff']:.3e} | {iou} | {tt_s} |")

    out = osp.join(HERE, "ablation_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out} and ablation_summary.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
