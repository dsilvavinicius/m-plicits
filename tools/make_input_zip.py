"""Build m-plicits-data.zip — the single reproduction-input archive.

Collects, from the private working tree, exactly what data/README.md
documents: clean inputs, corrected noise inputs, outlier clouds, and the
released checkpoints; writes MANIFEST.txt + SHA256SUMS.txt; zips with paths
relative to the repository root so the archive unpacks in place.

Maintainer script: it runs against the private tree layout (i3d_work), not
against this repository. Kept here for provenance of what the archive
contains.

Usage:
    python tools/make_input_zip.py --src <path-to-i3d_work> --out m-plicits-data.zip
"""
import argparse
import hashlib
import os
import re
import os.path as osp
import zipfile

CHECKPOINT_STAGES = ("coarse", "medium", "fine")


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect(src):
    """Yield (abs_path, arc_path) pairs."""
    inp = osp.join(src, "data", "normalized_sphere", "input")
    for f in sorted(os.listdir(inp)):
        if f.endswith(".ply"):
            yield osp.join(inp, f), f"data/normalized_sphere/input/{f}"
    noise = osp.join(src, "data", "normalized_noise", "noise_data")
    for f in sorted(os.listdir(noise)):
        if f.endswith(".ply"):
            yield osp.join(noise, f), f"data/normalized_noise/noise_data/{f}"
    outl = osp.join(src, "rebuttal_2026", "outliers")
    if osp.isdir(outl):
        for f in sorted(os.listdir(outl)):
            if f.endswith(".ply"):
                yield osp.join(outl, f), f"appendix_experiments/outliers/{f}"
    # Checkpoints. The private tree has a few shapes split across "<shape>/" and
    # "<shape>.ply/" folders; the archive uses one "<shape>/" per shape, with
    # the plain folder taking precedence stage by stage.
    res = osp.join(src, "results")
    by_stem = {}
    for d in sorted(os.listdir(res)):
        if osp.isdir(osp.join(res, d)):
            stem = d[:-4] if d.endswith(".ply") else d
            by_stem.setdefault(stem, []).append(d)
    for stem, dirs in sorted(by_stem.items()):
        for stage in CHECKPOINT_STAGES:
            for name in ("best.pth", "config.yaml"):
                for d in sorted(dirs, key=lambda x: x.endswith(".ply")):
                    p = osp.join(res, d, stage, name)
                    if osp.isfile(p):
                        yield p, f"results/{stem}/{stage}/{name}"
                        break


def collect_extra_tree(root, arc_prefix, skip_names=(".DS_Store",), skip_dirs=("__MACOSX",), skip_ext=(".zip",)):
    """Renderer/attribute asset trees, junk filtered."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for f in sorted(filenames):
            if f in skip_names or osp.splitext(f)[1] in skip_ext:
                continue
            p = osp.join(dirpath, f)
            rel = osp.relpath(p, root).replace(os.sep, "/")
            yield p, f"{arc_prefix}/{rel}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to the private i3d_work tree")
    ap.add_argument("--out", default="m-plicits-data.zip")
    ap.add_argument("--renderer-data", default=None,
                    help="path to the renderer checkpoint folder (src/cuda/data)")
    ap.add_argument("--attributes-data", default=None,
                    help="path to the attributes data folder")
    ap.add_argument("--attributes-shapenets", default=None,
                    help="path to the attributes shapeNets folder")
    ap.add_argument("--noise-meshes", default=None,
                    help="root of the paper's noise reconstructions (rec/noise_meshes); written to a "
                         "second, optional archive m-plicits-paper-meshes.zip as paper_meshes/noise/<shape>__<level>.ply")
    args = ap.parse_args()

    if args.noise_meshes:
        pairs = []
        for dirpath, _, files in os.walk(args.noise_meshes):
            for f in sorted(files):
                m = re.match(r"(.+)_noise_1_0_(coarse|medium|fine)\.ply$", f)
                if m:
                    pairs.append((osp.join(dirpath, f), f"paper_meshes/noise/{m.group(1)}__{m.group(2)}.ply"))
        write_zip(osp.join(osp.dirname(osp.abspath(args.out)), "m-plicits-paper-meshes.zip"), pairs,
                  "m-plicits-paper-meshes.zip: the paper's reconstructions of the 1% noisy inputs "
                  "(Tab. 3 ours rows), for `python reproduce.py paper-noise`.")

    pairs = list(collect(args.src))
    if args.renderer_data:
        pairs += list(collect_extra_tree(args.renderer_data, "renderer/cuda/data"))
    if args.attributes_data:
        pairs += list(collect_extra_tree(args.attributes_data, "attributes/data"))
    if args.attributes_shapenets:
        pairs += list(collect_extra_tree(args.attributes_shapenets, "attributes/shapeNets"))

    write_zip(args.out, pairs, "m-plicits-data.zip: unzip at the repository root of m-plicits.")


def write_zip(out, pairs, header):
    manifest, sums = [], []
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path, arc in pairs:
            sz = osp.getsize(path)
            total += sz
            z.write(path, arc)
            manifest.append(f"{sz:>12,d}  {arc}")
            sums.append(f"{sha256(path)}  {arc}")
            print(f"  + {arc} ({sz/1e6:.1f} MB)", flush=True)
        z.writestr("MANIFEST.txt", f"{header}\n{len(pairs)} files, {total/1e9:.2f} GB raw\n\n" + "\n".join(manifest) + "\n")
        z.writestr("SHA256SUMS.txt", "\n".join(sums) + "\n")
    print(f"\nwrote {out}: {len(pairs)} files, {total/1e9:.2f} GB raw, "
          f"{osp.getsize(out)/1e9:.2f} GB compressed; sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
