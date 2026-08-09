"""Screened Poisson WITHOUT the low-density trim (raw SPSR output).

Rationale: the density trim is SPSR's built-in outlier/low-confidence
rejection. For the outlier experiment we want an apples-to-apples comparison
in which NO method applies outlier rejection; and for the main comparison we
want to report SPSR's sensitivity to this choice explicitly.

Outputs (metered later by run_metrics.py with the paper protocol):
  meshes/<shape>__<cond>__spsrnotrim.ply           (10 shapes x clean,noise)
  outlier_meshes/<shape>__<rate>__spsrnotrim.ply   (outlier clouds)

Run: conda run -n new_i3d python run_spsr_notrim.py
"""
import glob
import os
import os.path as osp
import time

import open3d as o3d

HERE = osp.dirname(osp.abspath(__file__))
I3D = osp.dirname(HERE)
DEPTH = 10

STANFORD = ["normalized_armadillo_gt", "normalized_asian_dragon_gt",
            "normalized_lucy_gt", "normalized_thai_gt_half"]
THINGI = ["44234", "64764", "68381", "72870", "73075", "77245"]


def input_ply(shape, cond):
    if cond == "clean":
        return osp.join(I3D, "data", "normalized_sphere", "input", f"{shape}.ply")
    return osp.join(I3D, "data", "normalized_noise", "noise_data", f"{shape}.ply")


def spsr_notrim(in_ply, out_ply):
    mesh_in = o3d.io.read_triangle_mesh(in_ply)
    pc = o3d.geometry.PointCloud()
    pc.points = mesh_in.vertices
    if mesh_in.has_vertex_normals():
        pc.normals = mesh_in.vertex_normals
    else:
        pc.estimate_normals()
    t0 = time.time()
    mesh, _dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=DEPTH)
    # NO density trim applied - raw Poisson output.
    o3d.io.write_triangle_mesh(out_ply, mesh)
    return time.time() - t0


def main():
    times = []

    # --- main comparison shapes ---
    outdir = osp.join(HERE, "meshes")
    os.makedirs(outdir, exist_ok=True)
    for shape in STANFORD + THINGI:
        for cond in ("clean", "noise"):
            src = input_ply(shape, cond)
            if not osp.exists(src):
                print(f"[skip] {src}")
                continue
            out = osp.join(outdir, f"{shape}__{cond}__spsrnotrim.ply")
            if osp.exists(out):
                print(f"[cached] {osp.basename(out)}")
                continue
            dt = spsr_notrim(src, out)
            times.append((shape, cond, dt))
            print(f"{shape}/{cond}: no-trim SPSR in {dt:.1f}s", flush=True)

    # --- outlier clouds ---
    odir = osp.join(HERE, "outlier_meshes")
    os.makedirs(odir, exist_ok=True)
    for src in sorted(glob.glob(osp.join(HERE, "outliers", "*__o*.ply"))):
        base = osp.basename(src)[:-4]          # <shape>__<rate>
        out = osp.join(odir, f"{base}__spsrnotrim.ply")
        if osp.exists(out):
            print(f"[cached] {osp.basename(out)}")
            continue
        dt = spsr_notrim(src, out)
        times.append((base, "outlier", dt))
        print(f"{base}: no-trim SPSR in {dt:.1f}s", flush=True)

    with open(osp.join(HERE, "spsr_notrim_times.csv"), "w", encoding="utf-8") as f:
        f.write("shape,cond,seconds\n")
        for s, c, t in times:
            f.write(f"{s},{c},{t:.2f}\n")
    print("done - run run_metrics.py on meshes/ and outlier_meshes/")


if __name__ == "__main__":
    main()
