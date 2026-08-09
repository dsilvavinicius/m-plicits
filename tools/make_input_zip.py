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
    res = osp.join(src, "results")
    for shape in sorted(os.listdir(res)):
        for stage in CHECKPOINT_STAGES:
            for name in ("best.pth", "config.yaml"):
                p = osp.join(res, shape, stage, name)
                if osp.isfile(p):
                    yield p, f"results/{shape}/{stage}/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to the private i3d_work tree")
    ap.add_argument("--out", default="m-plicits-data.zip")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra 'abs_path:arc_path' pairs (renderer/attribute assets)")
    args = ap.parse_args()

    pairs = list(collect(args.src))
    for e in args.extra:
        a, _, arc = e.partition(":")
        pairs.append((a, arc))

    manifest, sums = [], []
    total = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path, arc in pairs:
            sz = osp.getsize(path)
            total += sz
            z.write(path, arc)
            manifest.append(f"{sz:>12,d}  {arc}")
            sums.append(f"{sha256(path)}  {arc}")
            print(f"  + {arc} ({sz/1e6:.1f} MB)", flush=True)
        z.writestr("MANIFEST.txt",
                   f"m-plicits-data.zip — {len(pairs)} files, {total/1e9:.2f} GB raw\n"
                   "Unzip at the repository root of m-plicits.\n\n" + "\n".join(manifest) + "\n")
        z.writestr("SHA256SUMS.txt", "\n".join(sums) + "\n")
    print(f"\nwrote {args.out}: {len(pairs)} files, {total/1e9:.2f} GB raw, "
          f"{osp.getsize(args.out)/1e9:.2f} GB compressed")


if __name__ == "__main__":
    main()
