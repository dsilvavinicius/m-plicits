#!/usr/bin/env python
# coding: utf-8

import open3d as o3d
import open3d.core as o3c
import torch
from plyfile import PlyData


def sample_on_surface(vertices: torch.Tensor, n_points: int, device: str):
    """Samples points in a torch tensor

    Parameters
    ----------
    vertices: torch.Tensor
        A mode-2 tensor where each row is a vertex.

    n_points: int
        The number of points to sample. If `n_points` == `vertices.shape[0]`,
        we simply return `vertices` without any change.

    device: str or torch.device
        The device where we should generate the indices of sampled points.

    Returns
    -------
    sampled: torch.tensor
        The points sampled from `vertices`. If
        `n_points` == `vertices.shape[0]`, then we simply return `vertices`.

    idx: torch.tensor
        The indices of points sampled from `vertices`. Naturally, these are
        row indices in `vertices`.

    See Also
    --------
    torch.randperm
    """
    if n_points == vertices.shape[0]:
        return vertices, torch.arange(start=0, end=n_points, device=device)
    idx = torch.randperm(vertices.shape[0], device=device)[:n_points]
    sampled = vertices[idx, ...]
    return sampled, idx


def calc_deltas(
    vertices: torch.tensor,
    scene: o3d.t.geometry.RaycastingScene,
    max_delta: float = 0.05,
    n_iters: int = 100,
    device: torch.device = torch.device("cpu"),
):
    vertices_int = vertices.detach().clone()
    eps = 2e-4

    with torch.no_grad():
        step_delta = max_delta / n_iters

        step_delta = step_delta * torch.ones(vertices_int.shape[0], device=device)
        deltas = step_delta * torch.ones(vertices_int.shape[0], device=device)  # + 0.01
        vertices_int = torch.hstack((vertices_int, deltas.unsqueeze(-1)))

        len = torch.sqrt(
            vertices_int[..., 3] ** 2
            + vertices_int[..., 4] ** 2
            + vertices_int[..., 5] ** 2
        )
        vertices_int[..., 3:6] /= len.unsqueeze(-1)

        i = 0
        while i < n_iters and torch.max(deltas) < max_delta:
            displaced_v = (
                vertices[..., :3] + deltas.unsqueeze(-1) * vertices_int[..., 3:6]
            )
            o3dpts = o3c.Tensor(displaced_v.cpu().numpy(), dtype=o3c.Dtype.Float32)
            sdf = torch.from_numpy(scene.compute_distance(o3dpts).numpy()).to(device)

            diff = torch.abs(vertices_int[..., -1] - sdf)
            step_delta = torch.where((diff) > eps, step_delta * 0.5, step_delta)
            deltas = torch.where(
                (diff) > eps,
                vertices_int[..., -1] - step_delta,
                vertices_int[..., -1] + step_delta,
            )

            vertices_int[..., -1] = deltas.clone()
            i += 1

    return vertices_int


def create_training_data_on_neighborhood(
    vertices: torch.tensor,
    n_on_surf: int,
    device: torch.device = torch.device("cpu"),
    delta=0.02,
    use_neigh_delta=False,
    dithering_sampling=True,
    delta_eps=0.0005,
):
    """Creates a set of training data with coordinates, normals and SDF
    values.

    Parameters
    ----------
    vertices: torch.tensor
        A mode-2 tensor with the mesh vertices.

    n_on_surf: int
        # of points to sample from the mesh.

    device: str or torch.device, optional
        The compute device where `vertices` is stored. By default its
        torch.device("cpu").

    delta: number, optional

    use_neigh_delta: boolean, optional

    dithering_sampling: boolean, optional

    Returns
    -------
    coords: dict[str => list[torch.Tensor]]
        A dictionary with points sampled from the surface (key = "on_surf")
        and the domain (key = "off_surf"). Each dictionary element is a list
        of tensors with the vertex coordinates as the first element of said
        list, the normals as the second element, finally, the SDF is the last
        element.

    See Also
    --------
    sample_on_surface
    """
    surf_pts, _ = sample_on_surface(vertices, n_on_surf, device=device)

    if use_neigh_delta:
        delta_neigh = surf_pts[..., -1:]
    else:
        delta_neigh = delta_eps * torch.ones_like(surf_pts[..., -1:])

    N = min(n_on_surf, vertices.shape[0])
    perturbed_pts = torch.rand((N, 3)).to(device)

    offset_pts_out = (
        surf_pts[..., :3] + delta_neigh * perturbed_pts[..., -1:] * surf_pts[..., 3:6]
    )

    pts = torch.cat((surf_pts[..., :3], offset_pts_out), dim=0)
    normals = torch.cat((surf_pts[..., 3:6], surf_pts[..., 3:6]), dim=0)
    sdfs = torch.cat(
        (
            torch.zeros_like(surf_pts[..., -1]),
            delta_neigh.squeeze(-1)
            * perturbed_pts[..., -1]
            * torch.ones_like(surf_pts[..., -1]),
        ),
        dim=0,
    )

    if dithering_sampling:
        perturbed_pts *= delta
        perturbed_pts += surf_pts[..., :3] + 0.5 * delta * surf_pts[..., 3:6]

        pts = torch.cat((pts, perturbed_pts), dim=0)
        normals = torch.cat((normals, surf_pts[..., 3:6]), dim=0)
        sdfs = torch.cat((sdfs, -1 * torch.ones_like(surf_pts[..., -1])), dim=0)

    return {"on_surf": [pts, normals, sdfs]}
