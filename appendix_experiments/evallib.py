"""Rebuttal ablation helpers.

CD / Hausdorff / IoU are computed by the paper's own metric code
(J:/projects/metrics/meshes/compute_distance_metrics.py, run via run_metrics.py
in the `metrics` conda env) - NOT here, so the rebuttal numbers use exactly the
protocol of the submission.

This module only holds the rebuttal-specific nesting-violation metric and small
utilities used by run_ablation.py inside the `new_i3d` env.
"""
import json

import torch


@torch.no_grad()
def nesting_violation(coarse, recon_mesh_path, delta1, device="cuda:0"):
    """Fraction of the FINAL reconstructed surface lying OUTSIDE the coarse
    delta_1 band: evaluate |f_1| at the reconstructed mesh vertices (points on
    the final zero-level set, up to marching-cubes interpolation) and report
    the fraction with |f_1| >= delta_1.

    These are exactly the surface regions that multiscale sphere tracing and
    the adaptive marching cubes would miss (false negatives), because both
    trust the band to contain the final surface (Eq. 3). Resolution-independent
    by construction (unlike a grid-band count).
    """
    import numpy as np
    import open3d as o3d

    v = np.asarray(o3d.io.read_triangle_mesh(recon_mesh_path).vertices,
                   dtype=np.float32)
    if v.shape[0] == 0:
        return {"surface_pts": 0, "violations": 0, "violation_rate": None}
    vt = torch.from_numpy(v).to(device)
    violated = 0
    for i in range(0, vt.shape[0], 2_000_000):
        f1 = coarse(vt[i : i + 2_000_000])["model_out"].abs().squeeze(-1)
        violated += int((f1 >= delta1).sum())
    return {
        "surface_pts": int(vt.shape[0]),
        "violations": violated,
        "violation_rate": violated / vt.shape[0],
    }


@torch.no_grad()
def mc_select(models, deltas, filename, N=512, device="cuda:0", max_batch=32**3):
    """Marching cubes for CHAINED STANDALONE SDFs (ablation cell c).

    models = [f1, f2, f3] are independent SDFs; f_{i+1} is only valid inside
    the band |f_i| < deltas[i]. The grid value is f1 outside the first band,
    f2 inside it (but outside the second), f3 innermost — SELECT, not sum.
    Returns the sampling time in seconds.
    """
    import time as _time

    from i3d.meshing import (convert_sdf_samples_to_ply,
                             gen_mc_coordinate_grid, save_ply)

    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    samples = gen_mc_coordinate_grid(N, voxel_size, device=device)
    num = N ** 3
    head = 0
    t0 = _time.perf_counter()
    while head < num:
        pts = samples[head : head + max_batch, :3]
        sdf = models[0](pts)["model_out"].squeeze(-1)
        for i in range(1, len(models)):
            idx = sdf.abs() < deltas[i - 1]
            if not idx.any():
                break
            sdf[idx] = models[i](pts[idx])["model_out"].squeeze(-1)
        samples[head : head + max_batch, 3] = sdf
        head += max_batch
    dt = _time.perf_counter() - t0

    vals = samples[:, 3].reshape(N, N, N).detach().cpu()
    verts, faces, _, _ = convert_sdf_samples_to_ply(
        vals, voxel_origin, voxel_size, None, None)
    save_ply(verts, faces, filename)
    return dt


@torch.no_grad()
def max_abs_at_points(models_sum, pts_t, device="cuda:0"):
    """max_j |sum_i f_i(x_j)| over the given points, chunked."""
    out = 0.0
    for i in range(0, pts_t.shape[0], 2_000_000):
        chunk = pts_t[i : i + 2_000_000]
        s = torch.zeros(chunk.shape[0], 1, device=device)
        for m in models_sum:
            s += m(chunk)["model_out"]
        out = max(out, float(s.abs().max()))
    return out


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
