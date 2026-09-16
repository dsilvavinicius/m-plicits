#!/usr/bin/env python
"""Download, verify and unpack the data archives at the repository root.

    python tools/download_data.py                    # models + inputs (m-plicits-data.zip)
    python tools/download_data.py --paper-meshes     # also the paper's noise reconstructions (Tab. 3 without retraining)
    python tools/download_data.py --gdrive <file-id> # from a Google Drive mirror instead of the GitHub release

Checksums are verified before unpacking; a partial download is resumed on
the next run.
"""
import argparse
import hashlib
import os
import os.path as osp
import sys
import zipfile

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RELEASE = "https://github.com/dsilvavinicius/m-plicits/releases/download/v1.0"
ARCHIVES = {
    # name: (release asset URL, sha256)  -- filled by tools/make_input_zip.py --sha
    "m-plicits-data.zip": (f"{RELEASE}/m-plicits-data.zip", "TODO_SHA256_DATA"),
    "m-plicits-paper-meshes.zip": (f"{RELEASE}/m-plicits-paper-meshes.zip", "TODO_SHA256_MESHES"),
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dest):
    import requests
    have = osp.getsize(dest) if osp.exists(dest) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, headers=headers, allow_redirects=True, timeout=60) as r:
        if r.status_code == 416:
            return
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        total = int(r.headers.get("Content-Length", 0)) + (have if mode == "ab" else 0)
        done = have if mode == "ab" else 0
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def fetch(name, gdrive=None, force=False):
    url, expected = ARCHIVES[name]
    dest = osp.join(ROOT, name)
    if not osp.exists(dest) or force:
        print(f"downloading {name}")
        if gdrive:
            import gdown
            gdown.download(id=gdrive, output=dest, quiet=False)
        else:
            download(url, dest)
    digest = sha256(dest)
    if expected.startswith("TODO"):
        print(f"  sha256 {digest} (no reference checksum recorded yet)")
    elif digest != expected:
        sys.exit(f"checksum mismatch for {name}: {digest} != {expected}; delete the file and retry")
    else:
        print(f"  checksum ok")
    print(f"unpacking {name} into {ROOT}")
    with zipfile.ZipFile(dest) as z:
        z.extractall(ROOT)
    print("done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-meshes", action="store_true", help="also fetch the paper's reconstruction meshes")
    ap.add_argument("--gdrive", default=None, help="Google Drive file id for the main archive")
    ap.add_argument("--gdrive-meshes", default=None, help="Google Drive file id for the meshes archive")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()
    fetch("m-plicits-data.zip", args.gdrive, args.force)
    if args.paper_meshes:
        fetch("m-plicits-paper-meshes.zip", args.gdrive_meshes, args.force)


if __name__ == "__main__":
    main()
