'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''

import numpy as np
import plyfile
from skimage.measure import marching_cubes
import time
import torch


def gen_mc_coordinate_grid(N: int, voxel_size: float, t: float = None,
                           device: str = "cpu",
                           voxel_origin: list = [-1, -1, -1]) -> torch.Tensor:
    """Creates the coordinate grid for inference and marching cubes run.

    Parameters
    ----------
    N: int
        Number of elements in each dimension. Total grid size will be N ** 3

    voxel_size: number
        Size of each voxel

    t: float, optional
        Reconstruction time. Required for space-time models. Default value is
        None, meaning that time is not a model parameter

    device: string, optional
        Device to store tensors. Default is CPU

    voxel_origin: list[number, number, number], optional
        Origin coordinates of the volume. Must be the (bottom, left, down)
        coordinates. Default is [-1, -1, -1]

    Returns
    -------
    samples: torch.Tensor
        A (N**3, 3) shaped tensor with samples' coordinates. If t is not None,
        then the return tensor is has 4 columns instead of 3, with the last
        column equalling `t`.
    """
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())

    sdf_coord = 3
    if t is not None:
        sdf_coord = 4

    # (x,y,z,sdf) if we are not considering time
    # (x,y,z,t,sdf) otherwise
    samples = torch.zeros(N ** 3, sdf_coord + 1, device=device,
                          requires_grad=False)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    # adding the time
    if t is not None:
        samples[:, sdf_coord-1] = t

    return samples


def create_mesh(
    decoder,
    filename="",
    t=None,
    N=256,
    max_batch=64 ** 3,
    offset=None,
    scale=None,
    device="cpu",
    silent=False
):
    decoder.eval()
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not
    # the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    samples = gen_mc_coordinate_grid(N, voxel_size, t=None if t == -1 else t,
                                     device=device)

    sdf_coord = 3 if t is None else 4

    num_samples = N ** 3
    head = 0

    start = time.time()
    with torch.no_grad():
        while head < num_samples:
            sample_subset = samples[head:min(head + max_batch, num_samples),
                                    0:sdf_coord]

            samples[head:min(head + max_batch, num_samples), sdf_coord] = \
                decoder(sample_subset)["model_out"].squeeze()

            head += max_batch

    end = time.time()
    sdf_values = samples[:, sdf_coord]
    sdf_values = sdf_values.reshape(N, N, N).detach().cpu()

    if not silent:
        print(f"Sampling took: {end-start} s")

    verts, faces, normals, values = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )

    if filename:
        if not silent:
            print(f"Saving mesh to {filename}")

        save_ply(verts, faces, filename)

        if not silent:
            print("Done")

    return verts, faces, normals, values


def create_mesh_multistage(
    decoders,
    deltas=[],
    filename="",
    t=None,
    N=256,
    max_batch=64 ** 3,
    offset=None,
    scale=None,
    device="cpu",
    silent=False
):
    """Evaluates the `decoders` on a 3D grid with `N^3` points.

    Parameters
    ----------
    decoders: Collection[i3d.model.SIREN]
        Ordered list of decoders. The first element is treated as a coarse
        network, while the remaining elements are residuals, whose outputs are
        added to the results of `decoders[0]`.

    deltas: List[float], optional
        Minimum values to evaluate the finer stages in `decoders`. Note that
        this list should be either empty or `len(deltas) = len(decoders) - 1`.
        If this list is empty (default) and len(decoders) > 1, then we
        evaluate all decoders for all grid elements.

    filename: str, optional
        Filename to save the resulting PLY. Default is empty, meaning that the
        file will not be saved.

    t: float, optional
        If `decoders` is an \mathbb{R}^4 \rarrow \mathbb{R} network, `t` is the
        parameter value to append to all points in the generated grid. By
        default is `None` meaning no parameter value.

    N: int, optional
        Number of voxels per side of the input grid. The total number of grid
        elements is `N**3`. Default value is 256, meaning that the total grid
        size is 256^3 cells.

    max_batch: int, optional
        Number of elements feed as input to `decoders` at each call. Optional,
        default value is 64^3.

    offset: float, optional
    scale: float, optional
    device: str, torch.device, optional
    silent: boolean, optional

    Returns
    -------
    verts:
    faces:
    normals:
    values:
    sampling_time:

    See Also
    --------
    convert_sdf_samples_to_ply, gen_mc_coordinate_grid
    """
    decoders_copy = [d.eval() for d in decoders]
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not
    # the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    samples = gen_mc_coordinate_grid(N, voxel_size, t=None if t == -1 else t,
                                     device=device)

    sdf_coord = 3 if t is None else 4
    num_samples = N ** 3
    head = 0

    start_sampling = time.perf_counter()

    with torch.no_grad():
        while head < num_samples:
            sample_subset = samples[head:min(head + max_batch, num_samples),
                                    0:sdf_coord]

            # Base level SDF evaluation
            sdf_base = decoders_copy[0](sample_subset)["model_out"].squeeze()
            torch.cuda.current_stream().synchronize()

            for i in range(1, len(decoders_copy)):
                if deltas:
                    idx = torch.abs(sdf_base) < deltas[i - 1]
                    if not idx.any():
                        # If on level i there are no points with
                        # abs(f) < delta_{i-1}, then its unlikely that there will
                        # be points on level i+1 with abs(f) < delta_{i}, so we
                        # break the loop here.
                        break

                    sdf_base[idx] += decoders_copy[i](sample_subset[idx, ...])["model_out"].squeeze()
                else:
                    sdf_base += decoders_copy[i](sample_subset)["model_out"].squeeze()

            samples[head:min(head + max_batch, num_samples), sdf_coord] = sdf_base
            head += max_batch

    torch.cuda.current_stream().synchronize()

    end_sampling = time.perf_counter()
    sdf_values = samples[:, sdf_coord]
    sdf_values = sdf_values.reshape(N, N, N).detach().cpu()

    if not silent:
        print(f"Sampling took: {end_sampling-start_sampling} s")

    verts, faces, normals, values = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )

    if filename:
        if not silent:
            print(f"Saving mesh to {filename}")

        save_ply(verts, faces, filename)

        if not silent:
            print("Done")

    return verts, faces, normals, values, end_sampling-start_sampling


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    if isinstance(pytorch_3d_sdf_tensor, torch.Tensor):
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.detach().cpu().numpy()
    else:
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    # Check if the cubes contains the zero-level set
    level = 0.0
    if level < numpy_3d_sdf_tensor.min() or level > numpy_3d_sdf_tensor.max():
        print("Surface level must be within volume data range.")
    else:
        verts, faces, normals, values = marching_cubes(
            numpy_3d_sdf_tensor, level, spacing=[voxel_size] * 3
        )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    return mesh_points, faces, normals, values


def save_ply(
        verts: np.array,
        faces: np.array,
        filename: str,
        vertex_attributes: list = None
) -> None:
    """Converts the vertices and faces into a PLY format, saving the resulting
    file.

    Parameters
    ----------
    verts: np.array
        An NxD matrix with the vertices and its attributes (normals,
        curvatures, etc.). Note that we expect verts to have at least 3
        columns, each corresponding to a vertex coordinate.

    faces: np.array
        An Fx3 matrix with the vertex indices for each triangle.

    filename: str
        Path to the output PLY file.

    vertex_attributes: list of tuples
        A list with the dtypes of vertex attributes other than coordinates.

    Examples
    --------
    > # This creates a simple triangle and saves it to a file called
    > #"triagle.ply"
    > verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    > faces = np.array([[0, 1, 2]])
    > save_ply(verts, faces, "triangle.ply")

    > # Writting normal information as well
    > verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    > faces = np.array([[0, 1, 2]])
    > normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    > attrs = [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    > save_ply(verts, faces, "triangle_normals.ply", vertex_attributes=attrs)
    """
    # try writing to the ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    dtypes = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if vertex_attributes is not None:
        dtypes[3:3] = vertex_attributes

    verts_tuple = np.zeros(
        (num_verts,),
        dtype=dtypes
    )

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(
        faces_building,
        dtype=[("vertex_indices", "i4", (3,))]
    )

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(filename)


if __name__ == "__main__":
    import os
    import os.path as osp
    from i3d.model import from_pth

    voxels = 512
    batch_size = 100**3
    n_runs = 10

    test_configs = {
        "armadillo": {
            "coarse": osp.join(os.getcwd(), "results", "armadillo_coarse", "best.pth"),
            "medium": osp.join(os.getcwd(), "results", "arm_res_on_neigh_medium", "best.pth"),
            "fine": osp.join(os.getcwd(), "results", "arm_res_on_neigh_fine", "best.pth"),
            "original": osp.join(os.getcwd(), "results", "armadillo_siren", "best.pth"),
        },
        "lucy": {
            "coarse": osp.join(os.getcwd(), "results", "lucy_coarse", "best.pth"),
            "medium": osp.join(os.getcwd(), "results", "lucy_medium", "best.pth"),
            "fine": osp.join(os.getcwd(), "results", "lucy_fine", "best.pth"),
            "original": osp.join(os.getcwd(), "results", "lucy_siren", "best.pth"),
        },
        "asian dragon": {
            "coarse": osp.join(os.getcwd(), "results", "asian_dragon_coarse", "best.pth"),
            "medium": osp.join(os.getcwd(), "results", "asian_dragon_on_neigh_medium", "best.pth"),
            "fine": osp.join(os.getcwd(), "results", "asian_dragon_on_neigh_fine", "best.pth"),
            "original": osp.join(os.getcwd(), "results", "asian_dragon_siren", "best.pth"),
        },
        "thai statue": {
            "coarse": osp.join(os.getcwd(), "results", "thai_coarse", "best.pth"),
            "medium": osp.join(os.getcwd(), "results", "thai_on_neigh_medium", "best.pth"),
            "fine": osp.join(os.getcwd(), "results", "thai_on_neigh_fine", "best.pth"),
            "original": osp.join(os.getcwd(), "results", "thai_statue_siren", "best.pth"),
        }
    }

    for config_name, config in test_configs.items():
        print(f"################ Running for {config_name} ################")
        original = from_pth(config["original"], device="cuda:0")

        levels = [
            from_pth(config["coarse"], device="cuda:0"),
            from_pth(config["medium"], device="cuda:0"),
            from_pth(config["fine"], device="cuda:0")
        ]

        multistage_times = [None] * n_runs
        naive_times = [None] * n_runs
        coarse_times = [None] * n_runs
        original_times = [None] * n_runs

        deltas = [0.1, 0.05]

        # The first call serves just to put the structures is memory and check if
        # the PLY model is accurate.
        print("################### MULTISTAGE #####################")
        create_mesh_multistage(
            levels, deltas=deltas, filename=f"{config_name}_mip.ply",
            N=voxels, max_batch=batch_size, device="cuda:0", silent=True
        )

        for i in range(n_runs):
            _, _, _, _, sampling_time = create_mesh_multistage(
                levels, deltas, N=voxels, max_batch=batch_size,
                device="cuda:0", silent=True
            )
            multistage_times[i] = sampling_time

        # Naive multistage
        print("################### NAIVE MULTISTAGE #####################")
        create_mesh_multistage(
            levels, deltas=[], filename=f"{config_name}_mip_naive.ply",
            N=voxels, max_batch=batch_size, device="cuda:0", silent=True
        )
        for i in range(n_runs):
            _, _, _, _, sampling_time = create_mesh_multistage(
                levels, deltas=[], N=voxels, max_batch=batch_size,
                device="cuda:0", silent=True
            )
            naive_times[i] = sampling_time

        print("################### COARSE #####################")
        create_mesh_multistage(
            [levels[0]], deltas=[], filename=f"{config_name}_coarse.ply",
            N=voxels, max_batch=batch_size, device="cuda:0", silent=True
        )
        for i in range(n_runs):
            _, _, _, _, sampling_time = create_mesh_multistage(
                [levels[0]], deltas=[], N=voxels, max_batch=batch_size,
                device="cuda:0", silent=True
            )
            coarse_times[i] = sampling_time

        print("################### NO MULTISTAGE #####################")
        create_mesh_multistage(
            [original], deltas=[], filename=f"{config_name}_original.ply",
            N=voxels, max_batch=batch_size, device="cuda:0", silent=True
        )
        for i in range(n_runs):
            _, _, _, _, sampling_time = create_mesh_multistage(
                [original], deltas=[], N=voxels, max_batch=batch_size,
                device="cuda:0", silent=True
            )
            original_times[i] = sampling_time

        print("run;multistage_time_s;naive_multistage_time_s;coarse_time_s;original_time_s")
        for i in range(n_runs):
            print(f"{i};{multistage_times[i]};{naive_times[i]};{coarse_times[i]};{original_times[i]}")
