'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
#!/usr/bin/env python3

import logging
import numpy as np
from numpy.core.fromnumeric import shape, squeeze
import plyfile
from skimage.measure import marching_cubes
import time
import torch
import pytorch3d
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.vis.plotly_vis import plot_batch_individually    
from pytorch3d.io.ply_io import MeshPlyFormat
from iopath.common.file_io import PathManager
from pytorch3d.io import IO

import diff_operators


def create_mesh(
    decoder1, decoder2, filename, 
    N=256, max_batch=64 ** 3
):
    start = time.time()
    ply_filename = filename

    decoder1.eval()
    decoder2.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1.1, -1.1, -1.1]
    voxel_size = 2.2 / (N - 1)
    
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

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

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        model_in = sample_subset
        with torch.no_grad():
            decoder_sample_subset = decoder1(model_in)['model_out']

            samples[head : min(head + max_batch, num_samples), 3] = (
                decoder_sample_subset
                .squeeze()#.squeeze(1)
                .detach()
                .cpu()
            )
            head += max_batch

    sdf_values = samples[:, 3] 

    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply_with_texture(
        decoder1,
        decoder2,
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply"
    )

def eval_textures(decoder, coords):
    num_verts = coords.shape[0]
    # coords = torch.from_numpy(mesh_points).float().cuda()
    textures = []
    N = 200
    for i in range(N):
        coords_i = coords[int(num_verts*i/N): int(num_verts*(i+1)/N),:]
        model_output_i = decoder(coords_i.unsqueeze(0))

        tex_batch = model_output_i['model_out']#.squeeze(0)#.cpu().detach().numpy()
        if len(textures)==0:
            textures = tex_batch
        else:
            textures = torch.cat((textures, tex_batch), dim=1)
    return textures

def convert_sdf_samples_to_ply_with_texture(
    decoder1,
    decoder2,
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    
    verts, faces, normals, values = marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # computing the principal directions of mesh_points ----------------
    # pred_curvatures = compute_mesh_curvature_directions(decoder1, mesh_points)
    # print(pred_curvatures.shape)
    
    coords = torch.from_numpy(mesh_points).float().cuda()
    shapeNet = decoder1(coords)
    normal_coords = diff_operators.gradient(shapeNet['model_out'],shapeNet['model_in'])
    face_indexes = torch.from_numpy(faces.copy()).int().cuda()    
    
    with torch.no_grad():
        colors = eval_textures(decoder2, coords)
        # colors=decoder2(coords.unsqueeze(0))['model_out'].squeeze(0)[None, ...]
    
    #TODO: The Pytorch3D visualizer seems to have a bug where color components near 0 result in artifacts. This clamp is a workaround.
    print(colors)
    colors = colors.clamp(1e-2,1)
    colors = torch.cat((colors, torch.ones_like(colors[...,-1:])),dim=-1)
    colors = colors[..., :3] # remove the alpha channel in case it is present

    texture = TexturesVertex(colors)

    # Create a pointclouds object with positions and colors.
    mesh = pytorch3d.structures.Meshes(verts=[coords.squeeze(0)], faces=[face_indexes], textures=texture, verts_normals=[normal_coords])
    
    #colors *= 255
    #texture = TexturesVertex(colors.to(torch.uint8))
    
    # Create a pointclouds object with positions and colors.
    #mesh = pytorch3d.structures.Meshes(verts=[coords.squeeze(0)], faces=[face_indexes], textures=texture, verts_normals=[normal_coords])
    #mesh.textures.to(torch.uint8)

    # Plot the point cloud using the default AxisArgs and PointcloudsVisualizer.
    #fig = plot_batch_individually([mesh], subplot_titles=["plot1"])
    #fig.show()

    print("Saving .ply")
    ply_format = MeshPlyFormat()
    path_manager = PathManager()
    ply_format.save(mesh, path=ply_filename_out, path_manager=path_manager, binary=False, colors_as_uint8=True)
    #IO().save_mesh(mesh, path=ply_filename_out, binary=False)
    #save_ply(ply_filename_out, verts_with_textures, mesh.faces_packed())