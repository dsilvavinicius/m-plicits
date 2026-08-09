import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.renderer import (
    FoVPerspectiveCameras, MeshRenderer, MeshRasterizer, BlendParams, NDCMultinomialRaysampler, PointsRasterizationSettings,
    PointsRenderer, PointsRasterizer, NormWeightedCompositor
)
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.structures import Meshes, Pointclouds
import trimesh
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_ray_positions(n_pixels, device="cuda:0"):
    """Initializes the ray positions according to the experiment.

    Parameters
    ----------
    n_pixels: int
        Viewport width in pixels. We assume a square viewport, thus, the
        resulting tensor will be shaped (n_pixels^2, 3).

    experiment_name: str
        The experiment name. May be any string. For "lucy*", "buddha*" and,
        "bunny*", we employ slightly different initialization mechanisms.

    device: str, optional
        The device to store our tensor. By default its the first CUDA
        capable GPU.

    Returns
    -------
    pos: torch.Tensor
        The initial ray position tensor with shape (n_pixels^2, 3).
    """
    pos = torch.zeros((n_pixels * n_pixels, 3), requires_grad=True,
                      device=torch.device(device))
    
    with torch.no_grad():
        x = torch.linspace(-1, 1, steps=n_pixels)
        xs, ys = torch.meshgrid(x, x, indexing='ij')

        pos[..., 0] = xs.reshape(-1, 1).squeeze()
        pos[..., 1] = ys.reshape(-1, 1).squeeze() +0.001
        pos[..., 2] = 0.6 * torch.ones_like(pos[..., 1])

    return pos


def init_ray_dir(device):
    """Initializes the ray directions according to the experiment.

    Parameters
    ----------
    experiment_name: str
        The experiment name. May be any string. For "lucy*", "buddha*" and,
        "bunny*", we employ slightly different initialization mechanisms.

    device: str, optional
        The device to store our tensor. By default its the first CUDA
        capable GPU.

    Returns
    -------
    dir: torch.Tensor
        The ray direction tensor with shape (1, 3).
    """
    dir = torch.tensor([0.0, 0.0, 1.0], device=device)

    return dir

def mse(image1, image2):
    return np.mean((image1 - image2)**2)

def get_camera_position(mesh, distance_scale=0.58):
    # Calculate the bounding box of the mesh
    bbox = mesh.get_bounding_boxes().squeeze(0)
    min_coords, max_coords = bbox[:, 0], bbox[:, 1]

    # Calculate the center and the size of the bounding box
    center = (min_coords + max_coords) / 2
    size = (max_coords - min_coords).norm().item()

    # Calculate the camera position
    camera_position = center.clone().detach()
    camera_position[2] += size * distance_scale

    return camera_position

def sphere_tracing(sdf_network, image_width, image_height, cameras, max_iterations=100, epsilon=1e-3):
    # Create a raysampler object
    #raysampler = NDCMultinomialRaysampler(image_width=image_width, image_height=image_height, n_pts_per_ray=1, min_depth=-1, max_depth=1)

    # Compute the ray origins and directions
    #ray_bundle = raysampler(cameras)
    #ray_origins, ray_directions = ray_bundle.origins, ray_bundle.directions

    # Initialize the point cloud
    #point_cloud = torch.zeros_like(ray_origins)

    point_cloud = init_ray_positions(image_width, device=device)
    ray_directions = init_ray_dir(device=device)
    
    for _ in range(max_iterations):
        # Evaluate the SDF at the current points
        inference = sdf_network(point_cloud)['model_out']
        sdf_values = inference.squeeze()
        
        # Update the point cloud using the SDF values
        point_cloud += sdf_values.unsqueeze(-1) * ray_directions
        
        # Check for convergence
        if torch.max(torch.abs(sdf_values)) < epsilon:
            break
    
    # Reshape the point_cloud tensor
    point_cloud = point_cloud.view(1, -1, 3)

    return point_cloud

import argparse
_ap = argparse.ArgumentParser(description="MSE between classically textured mesh renders and the neural texture (suppl. texture table).")
_ap.add_argument("--obj", default="./data/earth_hallison/Terra.obj", help="ground-truth textured mesh (.obj)")
_ap.add_argument("--recon", default="./results/terra/reconstructions/current.ply", help="colored reconstruction (.ply) from test_sdf_on_neural_surface.py")
_ap.add_argument("--sdf-pth", default="./shapeNets/terra_w0_16.pth")
_ap.add_argument("--sdf-w0", type=int, default=16)
_ap.add_argument("--tex-pth", default="./results/terra/checkpoints/model_final.pth")
_ap.add_argument("--tex-w0", type=int, default=120)
_args = _ap.parse_args()

with torch.no_grad():
    # Load the usual mesh from an OBJ file
    obj_file = _args.obj
    #obj_file = "./data/spot/spot/spot_tex.obj"
    mesh_usual = load_objs_as_meshes([obj_file], device=device)

    # Load the SIREN mesh from a PLY file
    ply_file = _args.recon
    #ply_file = "./results/spot/reconstructions/current.ply"
    trimesh_mesh = trimesh.load_mesh(ply_file)

    verts = torch.tensor(trimesh_mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(trimesh_mesh.faces, dtype=torch.int64)
    verts_colors = torch.tensor(trimesh_mesh.visual.vertex_colors[:, :3], dtype=torch.float32) / 255.0
    verts_colors = verts_colors.unsqueeze(0)
    textures_siren = TexturesVertex(verts_features=verts_colors.to(device))
    mesh_siren = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures_siren)

    # Calculate the camera position for the SIREN mesh
    camera_position = get_camera_position(mesh_siren).to(device)

    # Update the camera position
    cameras = FoVPerspectiveCameras(device=device, R=torch.eye(3, device=device).unsqueeze(0), T=camera_position.unsqueeze(0))
    #cameras = FoVPerspectiveCameras(device=device)

    #cameras = FoVPerspectiveCameras(device=device).to(device)
    image_size = 1024
    lights = PointLights(device=device).to(device)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(image_size=image_size)
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)

    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=shader)

    siren_image = renderer(mesh_siren, cameras=cameras, lights=lights)[0, ..., :3].cpu().numpy()
    usual_mapping_image = renderer(mesh_usual, cameras=cameras, lights=lights)[0, ..., :3].cpu().numpy()

    # Calculate the MSE between the usual mesh image and the SIREN mesh image
    mse_usual_siren = mse(usual_mapping_image, siren_image)
    print("Mean Squared Error (Usual Mesh vs SIREN Mesh):", mse_usual_siren)

    # SPHERE TRACING
    # Load both SIREN models from .pth files
    shape_net_path = _args.sdf_pth
    shape_net_w0 = _args.sdf_w0
    tex_net_path = _args.tex_pth
    tex_net_w0 = _args.tex_w0

    # Initialize SIREN models and load the state dictionaries
    sdf_network = utils.from_pth(shape_net_path, device=device, w0=shape_net_w0)
    sdf_network.cuda()
    tex_network = utils.from_pth(tex_net_path, device=device, w0=tex_net_w0)
    tex_network.cuda()

    # Perform sphere tracing on the neural SDF to obtain a point cloud representing the 3D surface
    point_cloud = sphere_tracing(sdf_network, image_size, image_size, cameras).to(device)

    # For each point in the point cloud, use the texture neural network to obtain its corresponding color
    colors = tex_network(point_cloud)['model_out']

    # Create a Pointclouds object with the point cloud and colors
    pc = Pointclouds(point_cloud, features=colors.view(1, -1, 3))

    # Use the SoftPhongShader with the same lighting model for rendering the point cloud
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    #renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=shader)

    # Create a PointsRasterizationSettings object
    point_raster_settings = PointsRasterizationSettings(image_size=image_size, radius=0.005, points_per_pixel=1)

    # Create a PointsRasterizer object
    points_rasterizer = PointsRasterizer(cameras=cameras, raster_settings=point_raster_settings)

    # Create a NormWeightedCompositor object
    compositor = NormWeightedCompositor()

    # Create a PointsRenderer object with the PointsRasterizer and NormWeightedCompositor
    points_renderer = PointsRenderer(rasterizer=points_rasterizer, compositor=compositor)

    # Render the neural SDF point cloud
    neural_sdf_image = points_renderer(pc, cameras=cameras, lights=lights)[0, ..., :3].cpu().detach().numpy()

    # Render the point cloud
    #neural_sdf_image = renderer(pc, cameras=cameras, lights=lights)[0, ..., :3].cpu().numpy()

    mse_usual_neural_sdf = mse(usual_mapping_image, neural_sdf_image)
    print("Mean Squared Error (Usual Mesh vs Neural SDF):", mse_usual_neural_sdf)

    # Display all three images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(usual_mapping_image)
    axes[0].set_title('Usual Mesh Image')
    axes[0].axis('off')
    axes[1].imshow(siren_image)
    axes[1].set_title('SIREN Mesh Image')
    axes[1].axis('off')
    axes[2].imshow(neural_sdf_image)
    axes[2].set_title('Neural SDF Image')
    axes[2].axis('off')
    plt.show(block=True)