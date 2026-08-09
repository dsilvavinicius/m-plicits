# coding: utf-8

from cmath import sqrt
import math
from mesh_to_sdf import (get_surface_point_cloud, scale_to_unit_cube,
                         scale_to_unit_sphere)
import numpy as np
import torch
import trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure
import diff_operators
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pytorch3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.vis.plotly_vis import plot_batch_individually

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)

        k_range = 5000
        n = point_cloud.shape[0]
        point_cloud = point_cloud[np.absolute(point_cloud[:, 3]) <= k_range]
        out_min_k1 = n - point_cloud.shape[0]

        point_cloud = point_cloud[np.absolute(point_cloud[:, 4]) <= k_range]
        out_min_k2 = n - point_cloud.shape[0] + out_min_k1

        print(f"[WARN] Removed {out_min_k1} with abs(k1) > {k_range}.")
        print(f"[WARN] Removed {out_min_k2} with abs(k2) > {k_range}.")

        print("Finished loading point cloud")

        #exporting ply (point, curvature, normal):  x, y, z, k, nx, ny, nz
        #coords = point_cloud[:, :3]
        #curvatures = point_cloud[:, 3]
        #self.normals = point_cloud[:, 4:7]

        #exporting ply (point, curvatures, normal):  x, y, z, k1, k2, nx, ny, nz
        point_cloud = point_cloud[np.absolute(point_cloud[:, 3]) < 10000]
        point_cloud = point_cloud[np.absolute(point_cloud[:, 4]) < 10000]

        coords = point_cloud[:, :3]
        min_curvatures = point_cloud[:, 4]
        max_curvatures = point_cloud[:, 3]
        self.normals = point_cloud[:, 5:8]

        #for mesh lab curvatures
        #curvatures = point_cloud[:, 6]
        #self.normals = point_cloud[:, 3:6]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.min_curvatures = min_curvatures
        self.max_curvatures = max_curvatures

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]
        on_surface_min_curvature = self.min_curvatures[rand_idcs]
        on_surface_max_curvature = self.max_curvatures[rand_idcs]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1
        off_surface_min_curvature = np.zeros((off_surface_samples))
        off_surface_max_curvature = np.zeros((off_surface_samples))
        # We consider the curvature of the sphere centered in the origin with radius equal to the norm of the coordinate.
        # off_surface_curvature = 1 / (np.linalg.norm(off_surface_coords, axis=1) ** 2)

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)
        min_curvature = np.concatenate((on_surface_min_curvature, off_surface_min_curvature))
        min_curvature = np.expand_dims(min_curvature, -1)
        max_curvature = np.concatenate((on_surface_max_curvature, off_surface_max_curvature))
        max_curvature = np.expand_dims(max_curvature, -1)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float(),
                                                              'min_curvature': torch.from_numpy(min_curvature).float(),
                                                              'max_curvature': torch.from_numpy(max_curvature).float()}


class PointCloudTubular(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        #exporting ply (point, curvatures, normal):  x, y, z, k1, k2, nx, ny, nz
        coords = point_cloud[:, :3]
        #min_curvatures = point_cloud[:, 4]
        #max_curvatures = point_cloud[:, 3]
        self.normals = point_cloud[:, 5:8]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        #self.min_curvatures = min_curvatures
        #self.max_curvatures = max_curvatures

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points 
        in_surface_samples  = self.on_surface_points  
        out_surface_samples = self.on_surface_points 
        total_samples = in_surface_samples + self.on_surface_points + out_surface_samples + off_surface_samples
      
        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        #tubular vicinity
        epsilon = 0.0001
        in_surface_coords  = on_surface_coords - epsilon*on_surface_normals
        out_surface_coords = on_surface_coords + epsilon*on_surface_normals

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1
        # We consider the curvature of the sphere centered in the origin with radius equal to the norm of the coordinate.
        # off_surface_curvature = 1 / (np.linalg.norm(off_surface_coords, axis=1) ** 2)

        sdf = np.zeros((total_samples, 1))#on-surface = 0 
        sdf[in_surface_samples + self.on_surface_points + out_surface_samples:, :] = -1  # off-surface = -1
        sdf[in_surface_samples + self.on_surface_points : in_surface_samples +  self.on_surface_points + out_surface_samples , :] = epsilon  # out-surface = epsilon
        sdf[ : in_surface_samples , :] = -epsilon  # in-surface = -epsilon

        # coordinates of the neighborhood of the tubular vicinity + off_surface
        coords = np.concatenate((in_surface_coords, on_surface_coords), axis=0)
        coords = np.concatenate((coords, out_surface_coords), axis=0)
        coords = np.concatenate((coords, off_surface_coords), axis=0)

        # duplicate the normals
        normals = np.concatenate((on_surface_normals, on_surface_normals), axis=0)
        normals = np.concatenate((normals, on_surface_normals), axis=0)
        normals = np.concatenate((normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}


class PointCloudNonRandom(Dataset):
    """Point Cloud dataset where the sampling is done by a proper sampler
    instead of inside __getitem__. The goal is to decouple the dataset
    from the sampling strategy, allowing us to experiment with diferent
    strategies.

    Parameters
    ----------
    pointcloud_path: str
        Path to the input file. This file is loaded as a numpy array. We assume
        that the data is organized as follows: x, y, z, nx, ny, nz, k1, k2, sdf

    keep_aspect_ratio: boolean, optional
        Indicates whether the mesh aspect ratio will be mantained when
        reshaping it to fit in a bounding box of size 2 (-1, 2). Default is
        True.

    k_range: int, optional
        The maximum curvature value to allow. Any samples with absolute value
        of curvature larger than this will be REMOVED from the cloud. Default
        value is 10000.

    See Also
    --------
    numpy.genfromtxt
    """
    def __init__(self, pointcloud_path, keep_aspect_ratio=True, k_range=10000):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        # exporting ply (point, curvatures, normal):  x, y, z, k1, k2, nx, ny, nz
        # Removing points with absurd curvatures.
        n = point_cloud.shape[0]
        point_cloud = point_cloud[np.absolute(point_cloud[:, 3]) <= k_range]
        out_min_k1 = n - point_cloud.shape[0]

        point_cloud = point_cloud[np.absolute(point_cloud[:, 4]) <= k_range]
        out_min_k2 = n - point_cloud.shape[0] + out_min_k1

        print(f"[WARN] Removed {out_min_k1} with abs(k1) > {k_range}.")
        print(f"[WARN] Removed {out_min_k2} with abs(k2) > {k_range}.")

        coords = point_cloud[:, :3]
        max_curvatures = point_cloud[:, 3]
        min_curvatures = point_cloud[:, 4]
        self.normals = point_cloud[:, 5:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.min_curvatures = min_curvatures
        self.max_curvatures = max_curvatures

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        if not isinstance(idx, list):
            idx = [idx]

        on_surface_coords = self.coords[idx, :]
        on_surface_normals = self.normals[idx, :]
        on_surface_min_curvature = self.min_curvatures[idx]
        on_surface_max_curvature = self.max_curvatures[idx]

        off_surface_coords = np.random.uniform(-1, 1, size=(len(idx), 3))
        off_surface_normals = np.ones((len(idx), 3)) * -1
        off_surface_min_curvature = np.zeros((len(idx)))
        off_surface_max_curvature = np.zeros((len(idx)))

        sdf = np.zeros((2 * len(idx), 1))  # on-surface = 0
        sdf[len(idx):, :] = -1  # off-surface = -1

        coords = np.vstack((on_surface_coords, off_surface_coords))
        normals = np.vstack((on_surface_normals, off_surface_normals))
        min_curvature = np.concatenate((on_surface_min_curvature, off_surface_min_curvature))
        min_curvature = np.expand_dims(min_curvature, -1)
        max_curvature = np.concatenate((on_surface_max_curvature, off_surface_max_curvature))
        max_curvature = np.expand_dims(max_curvature, -1)

        return {
            "coords": torch.from_numpy(coords).float()
        }, {
            "sdf": torch.from_numpy(sdf).float(),
            "normals": torch.from_numpy(normals).float(),
            "min_curvature": torch.from_numpy(min_curvature),
            "max_curvature": torch.from_numpy(max_curvature)
        }


class PointCloudTubularCurvatures(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)

        k_range = 5000

        n = point_cloud.shape[0]
        point_cloud = point_cloud[np.absolute(point_cloud[:, 3]) <= k_range]
        out_min_k1 = n - point_cloud.shape[0]

        point_cloud = point_cloud[np.absolute(point_cloud[:, 4]) <= k_range]
        out_min_k2 = n - point_cloud.shape[0] + out_min_k1

        print(f"[WARN] Removed {out_min_k1} with abs(k1) > {k_range}.")
        print(f"[WARN] Removed {out_min_k2} with abs(k2) > {k_range}.")

        print("Finished loading point cloud")

        #exporting ply (point, curvatures, normal):  x, y, z, k1, k2, nx, ny, nz
        coords = point_cloud[:, :3]
        min_curvatures = point_cloud[:, 4]
        max_curvatures = point_cloud[:, 3]
        self.normals = point_cloud[:, 5:8]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.min_curvatures = min_curvatures
        self.max_curvatures = max_curvatures

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points
        in_surface_samples = self.on_surface_points
        out_surface_samples = self.on_surface_points
        total_samples = in_surface_samples + self.on_surface_points + out_surface_samples + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]
        on_surface_min_curvature = np.expand_dims(self.min_curvatures[rand_idcs],-1)
        on_surface_max_curvature = np.expand_dims(self.max_curvatures[rand_idcs],-1)

        #tubular vicinity using curvature radius
        epsilon = 0.0005
        curvature_radius = 1./(np.maximum(np.absolute(on_surface_min_curvature), np.absolute(on_surface_max_curvature))) 
        curvature_radio = np.min(curvature_radius)
        curvature_radio = np.minimum(curvature_radio, epsilon)
       
        in_surface_coords  = on_surface_coords - curvature_radio*on_surface_normals
        out_surface_coords = on_surface_coords + curvature_radio*on_surface_normals

        in_surface_min_curvature, in_surface_max_curvature = diff_operators.principal_curvature_parallel_surface(on_surface_min_curvature,
                                                                                                                 on_surface_max_curvature, -curvature_radio)
        out_surface_min_curvature, out_surface_max_curvature = diff_operators.principal_curvature_parallel_surface(on_surface_min_curvature,
                                                                                                                   on_surface_max_curvature, curvature_radio)

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1
        off_surface_min_curvature = np.zeros_like(on_surface_min_curvature)
        off_surface_max_curvature = np.zeros_like(on_surface_max_curvature)
        # We consider the curvature of the sphere centered in the origin with radius equal to the norm of the coordinate.
        # off_surface_curvature = 1 / (np.linalg.norm(off_surface_coords, axis=1) ** 2)

        sdf = np.zeros((total_samples, 1))#on-surface = 0 
        sdf[in_surface_samples + self.on_surface_points + out_surface_samples:, :] = -1  # off-surface = -1
        sdf[in_surface_samples + self.on_surface_points : in_surface_samples +  self.on_surface_points + out_surface_samples , :] = curvature_radio  # out-surface = epsilon
        sdf[ : in_surface_samples , :] = -curvature_radio  # in-surface = -epsilon

        # coordinates of the neighborhood of the tubular vicinity + off_surface
        coords = np.concatenate((in_surface_coords, on_surface_coords), axis=0)
        coords = np.concatenate((coords, out_surface_coords), axis=0)
        coords = np.concatenate((coords, off_surface_coords), axis=0)

        # duplicate the normals
        normals = np.concatenate((on_surface_normals, on_surface_normals), axis=0)
        normals = np.concatenate((normals, on_surface_normals), axis=0)
        normals = np.concatenate((normals, off_surface_normals), axis=0)

        min_curvature = np.concatenate((in_surface_min_curvature, on_surface_min_curvature))
        min_curvature = np.concatenate((min_curvature, out_surface_min_curvature))
        min_curvature = np.concatenate((min_curvature, off_surface_min_curvature))
        #min_curvature = np.expand_dims(min_curvature, -1)

        max_curvature = np.concatenate((in_surface_max_curvature, on_surface_max_curvature))
        max_curvature = np.concatenate((max_curvature, out_surface_max_curvature))
        max_curvature = np.concatenate((max_curvature, off_surface_max_curvature))
        #max_curvature = np.expand_dims(max_curvature, -1)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float(),
                                                              'min_curvature': torch.from_numpy(min_curvature).float(),
                                                              'max_curvature': torch.from_numpy(max_curvature).float()}


class PointCloudSDF(Dataset):
    def __init__(self, mesh_path, xyz_path, scaling=None, batch_size=0, silent=False):
        super().__init__()
       
        # Loading the curvatures 
        print("Loading xyz point cloud")
        point_cloud_curv = np.genfromtxt(xyz_path)
       
        #exporting ply (point, normal, textures):  x, y, z, nx, ny, nz, r, g , b      
        coords = point_cloud_curv[:, :3]
        normals = point_cloud_curv[:, 3:6] #we should use mesh.vertex_normals because the sdf is computed using it
        
        #Loading the mesh
        self.input_path = mesh_path
        self.batch_size = batch_size

        print(f"Loading mesh \"{mesh_path}\".")

        mesh = trimesh.load(mesh_path)
        # if scaling is not None and scaling:
        #     if scaling == "bbox":
        #         mesh = scale_to_unit_cube(mesh)
        #     else:
        #         raise ValueError("Invalid scaling option.")

        self.mesh = mesh
        print("Creating point-cloud and acceleration structures.")

        self.point_cloud = get_surface_point_cloud(mesh, surface_point_method="scan", calculate_normals=True )

        #bbox scaling
        # vertices = point_cloud_curv[:, 0:3] - mesh.bounding_box.centroid
        # vertices *= 2 / np.max(mesh.bounding_box.extents)       

        self.surface_samples = torch.from_numpy(np.hstack((
            coords,
            normals,
            np.zeros((len(coords), 1))
        )).astype(np.float32))

        if not silent:
            print("Done preparing the dataset.")

    def __len__(self):
        lenght = self.surface_samples.size(0) // (self.batch_size) + 1
        return lenght

    def __getitem__(self, idx):
        return self._random_sampling(self.batch_size)

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.surface_samples.size(0)

        on_surface_count = n_points
        off_surface_count = n_points

        idx = np.random.choice(self.surface_samples.size(0), on_surface_count)
        on_surface_samples = self.surface_samples[idx, ...]
        
        off_surface_points = np.random.uniform(-1, 1, size=(off_surface_count, 3))
        off_surface_sdf, off_surface_normals = self.point_cloud.get_sdf(
            off_surface_points,
            use_depth_buffer=False,
            return_gradients=True
        )

        off_surface_samples = torch.from_numpy(np.hstack((
            off_surface_points,
            off_surface_normals,
            off_surface_sdf[:, np.newaxis]
        )).astype(np.float32))

        samples = torch.cat((on_surface_samples, off_surface_samples), dim=0)

        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": samples[:, :3].float()
        }, {
            "normals": samples[:, 3:6].float(),
            "sdf": samples[:, -1].unsqueeze(-1).float()
        }
        

class PointCloudSDFPreComputedCurvatures(Dataset):
    """Data class of a point-cloud that calculates the SDF values of point
    samples and schedules the samples by their curvatures.

    Parameters
    ----------
    mesh_path: str

    low_med_percentiles: collection[numbers], optional

    curvature_func: function(trimesh.Mesh, list[points], number), optional

    curvature_fracs: collection[numbers], optional

    scaling: str, optional

    batch_size: int, optional

    silent: boolean, optional

    See Also
    --------
    trimesh.curvature.discrete_gaussian_curvature_measure,
    trimesh.curvature.discrete_mean_curvature_measure
    """
    def __init__(self, mesh_path, low_med_percentiles=(70, 95),
                 curvature_func=discrete_gaussian_curvature_measure,
                 curvature_fracs=(0.5, 0.4, 0.1), scaling=None,
                 batch_size=0, silent=False):
        super().__init__()

        # Loading the curvatures 
        print("Loading xyz point cloud")
        point_cloud = np.genfromtxt('./data/armadillo_principal_curv.xyz')
        print("Finished loading point cloud")

        #exporting ply (point, curvatures, normal):  x, y, z, k1, k2, nx, ny, nz
        #coords = point_cloud[:, :3]
        min_curvatures = point_cloud[:, 4]
        max_curvatures = point_cloud[:, 3]
        #g_curvatures = min_curvatures*max_curvatures
        
        #check the signal
        g_curvatures = -0.5*(min_curvatures+max_curvatures)
        #normals = point_cloud[:, 5:8]

        #Loading the mesh
        self.input_path = mesh_path
        self.batch_size = batch_size

        if not silent:
            print(f"Loading mesh \"{mesh_path}\".")

        mesh = trimesh.load(mesh_path)
        if scaling is not None and scaling:
            if scaling == "bbox":
                mesh = scale_to_unit_cube(mesh)
            elif scaling == "sphere":
                mesh = scale_to_unit_sphere(mesh)
            else:
                raise ValueError("Invalid scaling option.")

        
        self.mesh = mesh
        if not silent:
            print("Creating point-cloud and acceleration structures.")

        self.point_cloud = get_surface_point_cloud(
            mesh,
            surface_point_method="scan",
            calculate_normals=True
        )

        scale = 2 / np.max(mesh.bounding_box.extents) 

        self.curvatures = g_curvatures*(1/scale)
        self.abs_curvatures = np.abs(self.curvatures)

        # low, medium, high curvature fractions
        self.curvature_fracs = curvature_fracs
        l1, l2 = np.percentile(self.abs_curvatures, low_med_percentiles)
        self.bin_edges = [
            np.min(self.abs_curvatures),
            l1,
            l2,
            np.max(self.abs_curvatures)
        ]

        self.surface_samples = torch.from_numpy(np.hstack((
            mesh.vertices.tolist(),
            mesh.vertex_normals,
            self.curvatures[:, np.newaxis],
            np.zeros((len(mesh.vertices), 1))
        )).astype(np.float32))

        print("Done preparing the dataset.")

    def __len__(self):
        return self.surface_samples.size(0) // self.batch_size

    def __getitem__(self, idx):
        return self._random_sampling(self.batch_size)

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.surface_samples.size(0)

        on_surface_count = n_points
        off_surface_count = n_points

        on_surface_sampled = 0
        low_curvature_pts = self.surface_samples[(self.abs_curvatures >= self.bin_edges[0]) & (self.abs_curvatures < self.bin_edges[1]), ...]
        low_curvature_idx = np.random.choice(
            range(low_curvature_pts.size(0)),
            size=int(math.floor(self.curvature_fracs[0] * on_surface_count)),
            replace=False
        )
        on_surface_sampled = len(low_curvature_idx)

        med_curvature_pts = self.surface_samples[(self.abs_curvatures >= self.bin_edges[1]) & (self.abs_curvatures < self.bin_edges[2]), ...]
        med_curvature_idx = np.random.choice(
            range(med_curvature_pts.size(0)),
            size=int(math.ceil(self.curvature_fracs[1] * on_surface_count)),
            replace=False
        )
        on_surface_sampled += len(med_curvature_idx)

        high_curvature_pts = self.surface_samples[(self.abs_curvatures >= self.bin_edges[2]) & (self.abs_curvatures <= self.bin_edges[3]), ...]
        high_curvature_idx = np.random.choice(
            range(high_curvature_pts.size(0)),
            size=on_surface_count - on_surface_sampled,
            replace=False
        )
        on_surface_samples = torch.cat((
            low_curvature_pts[low_curvature_idx, ...],
            med_curvature_pts[med_curvature_idx, ...],
            high_curvature_pts[high_curvature_idx, ...]
        ), dim=0)

        off_surface_points = np.random.uniform(-1, 1, size=(off_surface_count, 3))
        off_surface_sdf, off_surface_normals = self.point_cloud.get_sdf(
            off_surface_points,
            use_depth_buffer=False,
            return_gradients=True
        )
        off_surface_samples = torch.from_numpy(np.hstack((
            off_surface_points,
            off_surface_normals,
            np.zeros((off_surface_count, 1)),
            off_surface_sdf[:, np.newaxis]
        )).astype(np.float32))

        samples = torch.cat((on_surface_samples, off_surface_samples), dim=0)

        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": samples[:, :3].float()
        }, {
            "normals": samples[:, 3:6].float(),
            "curvature": samples[:, 6].unsqueeze(-1).float(),
            "sdf": samples[:, -1].unsqueeze(-1).float()
        }


def edgePlanarCornerSegmentation(surface_samples, on_surface_count, detector, bin_edges, proportions):
    on_surface_sampled = 0
    edge_pts = surface_samples[(detector >= bin_edges[0]) & (detector < bin_edges[1]), ...]
    edge_idx = np.random.choice(
        range(edge_pts.size(0)),
        size=int(math.floor(proportions[0] * on_surface_count)),
        replace=False
    )
    on_surface_sampled = len(edge_idx)

    planar_pts = surface_samples[(detector >= bin_edges[1]) & (detector < bin_edges[2]), ...]
    planar_idx = np.random.choice(
        range(planar_pts.size(0)),
        size=int(math.ceil(proportions[1] * on_surface_count)),
        replace=False
    )
    on_surface_sampled += len(planar_idx)

    corner_pts = surface_samples[(detector >= bin_edges[2]) & (detector <= bin_edges[3]), ...]
    corner_idx = np.random.choice(
        range(corner_pts.size(0)),
        size=on_surface_count - on_surface_sampled,
        replace=False
    )

    return torch.cat((
        edge_pts[corner_idx, ...],
        planar_pts[corner_idx, ...],
        corner_pts[corner_idx, ...]
    ), dim=0)


def lowMedHighCurvSegmentation(surface_samples, on_surface_count, abs_curvatures, bin_edges, proportions):
    on_surface_sampled = 0
    low_curvature_pts = surface_samples[(abs_curvatures >= bin_edges[0]) & (abs_curvatures < bin_edges[1]), ...]
    low_curvature_idx = np.random.choice(
        range(low_curvature_pts.size(0)),
        size=int(math.floor(proportions[0] * on_surface_count)),
        replace=False
    )
    on_surface_sampled = len(low_curvature_idx)

    med_curvature_pts = surface_samples[(abs_curvatures >= bin_edges[1]) & (abs_curvatures < bin_edges[2]), ...]
    med_curvature_idx = np.random.choice(
        range(med_curvature_pts.size(0)),
        size=int(math.ceil(proportions[1] * on_surface_count)),
        replace=False
    )
    on_surface_sampled += len(med_curvature_idx)

    high_curvature_pts = surface_samples[(abs_curvatures >= bin_edges[2]) & (abs_curvatures <= bin_edges[3]), ...]
    high_curvature_idx = np.random.choice(
        range(high_curvature_pts.size(0)),
        size=on_surface_count - on_surface_sampled,
        replace=False
    )
    
    return torch.cat((
        low_curvature_pts[low_curvature_idx, ...],
        med_curvature_pts[med_curvature_idx, ...],
        high_curvature_pts[high_curvature_idx, ...]
    ), dim=0)



class PointCloudSDFPreComputedCurvaturesDirections(Dataset):
    """Data class of a point-cloud that calculates the SDF values of point
    samples and schedules the samples by their curvatures.

    Parameters
    ----------
    mesh_path: str

    low_med_percentiles: collection[numbers], optional

    curvature_func: function(trimesh.Mesh, list[points], number), optional

    curvature_fracs: collection[numbers], optional

    scaling: str, optional

    uniform_sampling: boolean, optional

    batch_size: int, optional

    silent: boolean, optional

    See Also
    --------
    trimesh.curvature.discrete_gaussian_curvature_measure,
    trimesh.curvature.discrete_mean_curvature_measure
    """
    def __init__(self, mesh_path, xyz_path, low_med_percentiles=(70, 95),
                 curvature_fracs=(0.2, 0.6, 0.2), scaling=None,
                 uniform_sampling=False, batch_size=0, silent=False):
        super().__init__()
        self.uniform_sampling = uniform_sampling
        self.low_med_percentiles = low_med_percentiles

        # Loading the curvatures 
        print("Loading xyz point cloud")
        point_cloud_curv = np.genfromtxt(xyz_path)
        #point_cloud_curv = np.genfromtxt('./data/armadillo_curv_dir.xyz')
        #remove nan
        point_cloud_curv = point_cloud_curv[~np.isnan(point_cloud_curv[:, 4])] 
        print("Finished loading point cloud")

        #exporting ply (point, curvatures, normal, principal direction):  x, y, z, k1, k2, nx, ny, nz, kx, ky, kz       
        #coords = point_cloud[:, :3]
        min_curvatures = point_cloud_curv[:, 4]# the signal was changed
        max_curvatures = point_cloud_curv[:, 3]
        normals = point_cloud_curv[:, 5:8] #we should use mesh.vertex_normals because the sdf is computed using it
        max_dirs = point_cloud_curv[:, 8:11]

        #Loading the mesh
        self.input_path = mesh_path
        self.batch_size = batch_size

        print(f"Loading mesh \"{mesh_path}\".")

        mesh = trimesh.load(mesh_path)
        if scaling is not None and scaling:
            if scaling == "bbox":
                mesh = scale_to_unit_cube(mesh)
            else:
                raise ValueError("Invalid scaling option.")

        self.mesh = mesh
        print("Creating point-cloud and acceleration structures.")

        self.point_cloud = get_surface_point_cloud(mesh, surface_point_method="scan", calculate_normals=True )

        #self.diff_curvatures = np.abs(min_curvatures-max_curvatures)
        self.gauss_curvatures = min_curvatures*max_curvatures
        #self.abs_curvatures = 0.5*np.abs(min_curvatures+max_curvatures)
        self.abs_curvatures = np.abs(min_curvatures)+np.abs(max_curvatures)

        #using harris corner detector
        #self.abs_curvatures = min_curvatures*max_curvatures - 0.05*(min_curvatures+max_curvatures)**2

        # planar, edge, corner region fractions
        #self.curvature_fracs = curvature_fracs
        
        # low, medium, high curvature fractions
        self.curvature_fracs = curvature_fracs
        l1, l2 = np.percentile(self.abs_curvatures, low_med_percentiles)
        self.bin_edges = [
            np.min(self.abs_curvatures),
            l1,
            l2,
            np.max(self.abs_curvatures)
        ]

        #bbox scaling
        vertices = point_cloud_curv[:, 0:3] - mesh.bounding_box.centroid
        vertices *= 2 / np.max(mesh.bounding_box.extents)       

        self.surface_samples = torch.from_numpy(np.hstack((
            vertices,
            normals,
            min_curvatures[:, np.newaxis],
            max_curvatures[:, np.newaxis],
            max_dirs,
            np.zeros((len(vertices), 1))
        )).astype(np.float32))

        if not silent:
            print("Done preparing the dataset.")

    def __len__(self):
    #    return self.surface_samples.size(0) // self.batch_size

        lenght = self.surface_samples.size(0) // (self.batch_size) + 1
        # lenght = 17
        return lenght

        # # percentile of med curv samples times 1.2 used when we are using the whole dataset
        # p2o = 1.2*(self.low_med_percentiles[1]-self.low_med_percentiles[0])/100
        
        # # percentile of med curv samples times 1.2 used when we are using a percentile p2o/p2 of dataset
        # p2 = self.curvature_fracs[1]

        # lenght = int(math.floor((p2o/p2)*self.surface_samples.size(0))) // self.batch_size
        # return lenght
        # #return 100000 // self.batch_size
        #return 50000 // self.batch_size
        #return 10000 // self.batch_size

    def __getitem__(self, idx):
        return self._random_sampling(self.batch_size)

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.surface_samples.size(0)

        on_surface_count = n_points
        off_surface_count = n_points

        on_surface_samples = []

        if self.uniform_sampling:
            idx = np.random.choice(self.surface_samples.size(0), on_surface_count)
            on_surface_samples = self.surface_samples[idx, ...]
        else:
            #on_surface_samples=edgePlanarCornerSegmentation(self.surface_samples, on_surface_count, self.abs_curvatures, self.bin_edges, self.curvature_fracs)
            on_surface_samples = lowMedHighCurvSegmentation(self.surface_samples, on_surface_count, self.abs_curvatures, self.bin_edges, self.curvature_fracs)

        off_surface_points = np.random.uniform(-1, 1, size=(off_surface_count, 3))
        off_surface_sdf, off_surface_normals = self.point_cloud.get_sdf(
            off_surface_points,
            use_depth_buffer=False,
            return_gradients=True
        )

        off_surface_samples = torch.from_numpy(np.hstack((
            off_surface_points,
            off_surface_normals,
            np.zeros((off_surface_count, 1)),#min_curv
            np.zeros((off_surface_count, 1)),#max_curv
            np.ones((off_surface_count, 3)) * -1,#max_dirs
            off_surface_sdf[:, np.newaxis]
        )).astype(np.float32))

        samples = torch.cat((on_surface_samples, off_surface_samples), dim=0)

        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": samples[:, :3].float()
        }, {
            "normals": samples[:, 3:6].float(),
            "min_curvatures": samples[:, 6].unsqueeze(-1).float(),
            "max_curvatures": samples[:, 7].unsqueeze(-1).float(),
            "max_principal_directions": samples[:, 8:11].float(),
            "sdf": samples[:, -1].unsqueeze(-1).float()
        }


class PointCloudSDFTexture(Dataset):
    def __init__(self, trained_model, #mesh_path,
                 xyz_path, #scaling=None,
                 batch_size=0, silent=False):
        super().__init__()

        self.model = trained_model
        self.model.cuda()

        # Loading the curvatures 
        print("Loading xyz point cloud")
        point_cloud = np.genfromtxt(xyz_path)
        #point_cloud1 = np.genfromtxt('./data/egg_tensor.xyz') #TEST: point_cloud have a wrong scale
        
        #exporting ply (point, normal,texture):  x, y, z, nx, ny, nz, r, g, b       
        coords   = point_cloud[:,  :3]
        normals  = point_cloud[:, 3:6] #we should use mesh.vertex_normals because the sdf is computed using it
        textures = point_cloud[:, 6:9]/255 #coordinates in [0,1]

        #Loading the mesh
        #self.input_path = mesh_path
        self.batch_size = batch_size

        self.surface_samples = torch.from_numpy(np.hstack((
            coords,
            normals,
            textures
        )).astype(np.float32))

        if not silent:
            print("Done preparing the dataset.")

    def __len__(self):
        lenght = self.surface_samples.size(0) // (self.batch_size) + 1
        return lenght

    def __getitem__(self, idx):
        return self._random_sampling(self.batch_size)

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.surface_samples.size(0)

        on_surface_count = n_points
        
        on_surface_samples = []

        idx = np.random.choice(self.surface_samples.size(0), on_surface_count)
        on_surface_samples = self.surface_samples[idx, ...]
        
        samples = on_surface_samples

        #projecting the coords to the neural surface
        coords = samples[:, :3].float().cuda()
        trained_model = self.model(coords)
        coords = trained_model['model_in']
        sdf    = trained_model['model_out']

        grad = diff_operators.gradient(sdf, coords)
        # grad_norm = torch.norm(grad, dim=-1)
        # unit_grad = grad/grad_norm.unsqueeze(-1)
        coords = coords - sdf*grad
        coords = coords.cpu().detach()
        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": coords.float()
        }, {
            "normals": samples[:, 3:6].float(),
            "textures": samples[:, 6:9].float()
        }


class PointCloudMeshTexture(Dataset):
    def __init__(self, trained_model, mesh_path, batch_size=0, silent=False):
        super().__init__()

        self.shapeNet = trained_model
        self.shapeNet.cuda()

        self.meshes = load_objs_as_meshes(files=[mesh_path], device='cuda')
        self.batch_size = batch_size

        print("Done preparing the dataset.")

    def __len__(self):
        #lenght = self.surface_samples.size(0) // (self.batch_size) + 1
        return 1#lenght

    def __getitem__(self, idx):
        
        coords, normals, textures = sample_points_from_meshes(meshes=self.meshes, num_samples=self.batch_size, return_normals=True, return_textures=True)

        

        #projecting the coords to the neural surface
        trained_model = self.shapeNet(coords.cuda())
        coords = trained_model['model_in']
        sdf    = trained_model['model_out']

        grad = diff_operators.gradient(sdf, coords)
        grad_norm = torch.norm(grad, dim=-1)
        unit_grad = grad/grad_norm.unsqueeze(-1)
        coords = coords - sdf*unit_grad
        coords = coords.detach()
        
        # # Create a pointclouds object with positions and colors.
        # pointclouds = pytorch3d.structures.Pointclouds(points=[coords.squeeze(0)], features=[textures.squeeze(0)])

        # # Plot the point cloud using the default AxisArgs and PointcloudsVisualizer.
        # fig = plot_batch_individually([pointclouds], subplot_titles=["plot1"])
        # fig.show()

        return  {
                     "coords" : coords.float()
                }, {
                    "normals" : normals.float(),
                    "textures": textures.float()
                }
        #return self._random_sampling(self.batch_size)

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.surface_samples.size(0)

        on_surface_count = n_points
        
        on_surface_samples = []

        idx = np.random.choice(self.surface_samples.size(0), on_surface_count)
        on_surface_samples = self.surface_samples[idx, ...]
        
        samples = on_surface_samples

        #projecting the coords to the neural surface
        coords = samples[:, :3].float().cuda()
        trained_model = self.shapeNet(coords)
        coords = trained_model['model_in']
        sdf    = trained_model['model_out']

        grad = diff_operators.gradient(sdf, coords)
        # grad_norm = torch.norm(grad, dim=-1)
        # unit_grad = grad/grad_norm.unsqueeze(-1)
        coords = coords - sdf*grad
        coords = coords.cpu().detach()
        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": coords.float()
        }, {
            "normals": samples[:, 3:6].float(),
            "textures": samples[:, 6:9].float()
        }



class PointCloudSDFTexture4D(Dataset):
    def __init__(self, trained_model, mesh_path, xyz_path, scaling=None,
                 batch_size=0, silent=False):
        super().__init__()

        self.model = trained_model
        self.model.cuda()

        # Loading the curvatures 
        print("Loading xyz point cloud")
        point_cloud = np.genfromtxt(xyz_path)
        #point_cloud1 = np.genfromtxt('./data/egg_tensor.xyz') #TEST: point_cloud have a wrong scale
        
        #exporting ply (point, normal,texture):  x, y, z, nx, ny, nz, r, g, b       
        coords   = point_cloud[:,  :3]
        normals  = point_cloud[:, 3:6] #we should use mesh.vertex_normals because the sdf is computed using it
        textures = point_cloud[:, 6:9]/255 #coordinates in [0,1]

        #Loading the mesh
        self.input_path = mesh_path
        self.batch_size = batch_size


        mesh = trimesh.load(mesh_path)

        self.mesh = mesh
        print("Creating point-cloud and acceleration structures.")
        self.point_cloud = get_surface_point_cloud(mesh, surface_point_method="scan", calculate_normals=True )

        #bbox scaling
        coords = coords - mesh.bounding_box.centroid
        coords *= 2 / np.max(mesh.bounding_box.extents)       


        self.surface_samples = torch.from_numpy(np.hstack((
            coords,
            np.zeros((len(coords), 1)), #time zero
            normals,
            textures
        )).astype(np.float32))

        if not silent:
            print("Done preparing the dataset.")

    def __len__(self):
        lenght = self.surface_samples.size(0) // (self.batch_size) + 1
        return lenght

    def __getitem__(self, idx):
        return self._random_sampling(self.batch_size)
        #return self._random_sampling_with_tubular_neighborhood(self.batch_size)
        
    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.surface_samples.size(0)

        on_surface_count = n_points // 2
        off_surface_count = on_surface_count
        
        on_surface_samples = []

        idx = np.random.choice(self.surface_samples.size(0), on_surface_count)
        on_surface_samples = self.surface_samples[idx, ...]
      
        samples_on_time = torch.from_numpy(np.random.uniform(0, 1, size=(off_surface_count,1)).astype(np.float32))
        off_surface_samples = torch.cat((on_surface_samples[...,0:3], samples_on_time, on_surface_samples[...,4:10]), dim=-1)

        samples = torch.cat((on_surface_samples, off_surface_samples), dim=0 )

        #projecting the coords to the neural surface
        coords = samples[:, :4].float().cuda()
        trained_model = self.model(coords[:, :3])
        coords_3d = trained_model['model_in']
        sdf    = trained_model['model_out']

        grad = diff_operators.gradient(sdf, coords_3d)
        grad_norm = torch.norm(grad, dim=-1)
        unit_grad = grad/grad_norm.unsqueeze(-1)
        coords_3d = coords_3d - sdf*unit_grad
        #coords[...,0:3] = coords_3d

        coords = torch.cat((coords_3d, coords[...,3].unsqueeze(-1)), dim=-1 )

        coords = coords.cpu().detach()
        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": coords.float()
        }, {
            "normals": samples[:, 4:7].float(),
            "textures": samples[:, 7:10].float()
        }

    def _random_sampling_with_tubular_neighborhood(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.surface_samples.size(0)

        on_surface_count = n_points
        in_surface_count  = on_surface_count  
        out_surface_count = on_surface_count
        off_surface_count = in_surface_count + on_surface_count + out_surface_count


        on_surface_samples = []

        idx = np.random.choice(self.surface_samples.size(0), on_surface_count)
        on_surface_samples = self.surface_samples[idx, ...]
      
        #projecting the coords to the neural surface
        coords = on_surface_samples[:, :3].float().cuda()
        trained_model = self.model(coords)
        coords = trained_model['model_in']
        sdf = trained_model['model_out']

        grad = diff_operators.gradient(sdf, coords)
        grad_norm = torch.norm(grad, dim=-1)
        unit_grad = grad/grad_norm.unsqueeze(-1)
        coords = coords - sdf*unit_grad

        #update the gradient
        trained_model = self.model(coords)
        coords = trained_model['model_in']
        sdf = trained_model['model_out']
        grad = diff_operators.gradient(sdf, coords) 
        textures = diff_operators.mean_curvature(sdf, coords)

        #tubular neighborhood
        epsilon = 0.0001
        in_surface_coords  = coords - epsilon*grad
        out_surface_coords = coords + epsilon*grad

        coords = torch.cat((coords, in_surface_coords, out_surface_coords), dim=0 )
        textures = torch.cat((textures, textures, textures), dim=0 )
        
        coords = coords.cpu().detach()
        textures = textures.cpu().detach()
        
        on_surface_samples = torch.cat((coords, torch.zeros_like(textures)), dim=-1)

        #uniform sampling of time
        samples_on_time = torch.from_numpy(np.random.uniform(0, 1, size=(off_surface_count,1)).astype(np.float32))
        off_surface_samples = torch.cat((coords, samples_on_time), dim=-1)
        

        samples = torch.cat((on_surface_samples, off_surface_samples), dim=0 )
        textures = torch.cat((textures, -1*torch.ones_like(textures)), dim=0 )
 
        
        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": samples.float()
        }, {
            "textures": textures.float()
        }
        
class PointCloudSDFReacDiffusion(Dataset):
    def __init__(self, trained_model, mesh_path, xyz_path, scaling=None,
                 batch_size=0, silent=False):
        super().__init__()

        self.model = trained_model
        self.model.cuda()

        # Loading the curvatures 
        print("Loading xyz point cloud")
        point_cloud = np.genfromtxt(xyz_path)
        #point_cloud1 = np.genfromtxt('./data/egg_tensor.xyz') #TEST: point_cloud have a wrong scale
        
        #exporting ply (point, normal,texture):  x, y, z, nx, ny, nz, r, g, b       
        coords   = point_cloud[:,  :3]
        normals  = point_cloud[:, 3:6] #we should use mesh.vertex_normals because the sdf is computed using it
        textures = point_cloud[:, 6:9]/255 #coordinates in [0,1]

        #Loading the mesh
        self.input_path = mesh_path
        self.batch_size = batch_size

        mesh = trimesh.load(mesh_path)

        self.mesh = mesh
        print("Creating point-cloud and acceleration structures.")
        self.point_cloud = get_surface_point_cloud(mesh, surface_point_method="scan", calculate_normals=True )

        #bbox scaling
        coords = coords - mesh.bounding_box.centroid
        coords *= 2 / np.max(mesh.bounding_box.extents)       

        self.surface_samples = torch.from_numpy(np.hstack((
            coords,
            np.zeros((len(coords), 1)), #time zero
            normals,
            textures
        )).astype(np.float32))

        if not silent:
            print("Done preparing the dataset.")

    def __len__(self):
        lenght = self.surface_samples.size(0) // (self.batch_size) + 1
        return lenght

    def __getitem__(self, idx):
        return self._random_sampling(self.batch_size)
        #return self._random_sampling_with_tubular_neighborhood(self.batch_size)
        
    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.surface_samples.size(0)

        on_surface_count = n_points // 2
        off_surface_count = on_surface_count
        
        on_surface_samples = []

        idx = np.random.choice(self.surface_samples.size(0), on_surface_count)
        on_surface_samples = self.surface_samples[idx, ...]
      
        samples_on_time = torch.from_numpy(np.random.uniform(0, 1, size=(off_surface_count,1)).astype(np.float32))
        off_surface_samples = torch.cat((on_surface_samples[...,0:3], samples_on_time, on_surface_samples[...,4:10]), dim=-1)

        samples = torch.cat((on_surface_samples, off_surface_samples), dim=0 )

        #projecting the coords to the neural surface
        coords = samples[:, :4].float().cuda()
        trained_model = self.model(coords[:, :3])
        coords_3d = trained_model['model_in']
        sdf    = trained_model['model_out']

        grad = diff_operators.gradient(sdf, coords_3d)
        grad_norm = torch.norm(grad, dim=-1)
        unit_grad = grad/grad_norm.unsqueeze(-1)
        coords_3d = coords_3d - sdf*unit_grad
        #coords[...,0:3] = coords_3d

        coords = torch.cat((coords_3d, coords[...,3].unsqueeze(-1)), dim=-1 )

        coords = coords.cpu().detach()
        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": coords.float()
        }, {
            "normals": samples[:, 4:7].float(),
            "textures": samples[:, 7:10].float()
        }



class PointCloudSDFonSurface(Dataset):
    def __init__(self, trained_model, mesh_path, xyz_path, scaling=None,
                 batch_size=0, silent=False):
        super().__init__()

        self.model = trained_model
        self.model.cuda()

        print("Loading xyz point cloud")
        point_cloud = np.genfromtxt(xyz_path)
        
        #exporting ply (point, normal,texture):  x, y, z, nx, ny, nz, r, g, b       
        coords   = point_cloud[:,  :3]
        normals  = point_cloud[:, 3:6] #we should use mesh.vertex_normals because the sdf is computed using it
        distances  = point_cloud[:, 6] 

        #Loading the mesh
        self.input_path = mesh_path
        self.batch_size = batch_size

        mesh = trimesh.load(mesh_path)

        self.mesh = mesh
        print("Creating point-cloud and acceleration structures.")
        self.point_cloud = get_surface_point_cloud(mesh, surface_point_method="scan", calculate_normals=True )

        #bbox scaling
        coords = coords - mesh.bounding_box.centroid
        coords *= 2 / np.max(mesh.bounding_box.extents)       

        self.surface_samples = torch.from_numpy(np.hstack((
            coords,
            normals,
            distances[:, np.newaxis]
        )).astype(np.float32))

        if not silent:
            print("Done preparing the dataset.")

    def __len__(self):
        lenght = self.surface_samples.size(0) // (self.batch_size) + 1
        return lenght

    def __getitem__(self, idx):
        return self._random_sampling(self.batch_size)
        
    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.surface_samples.size(0)

        on_surface_count = n_points
        
        idx = np.random.choice(self.surface_samples.size(0), on_surface_count)
        samples = self.surface_samples[idx, ...]
      
        #projecting the coords to the neural surface
        coords = samples[:, :3].float().cuda()
        model_input = {"coords": coords}
        trained_model = self.model(model_input)
        coords_3d = trained_model['model_in']
        sdf = trained_model['model_out']

        grad = diff_operators.gradient(sdf, coords_3d)
        grad_norm = torch.norm(grad, dim=-1)
        unit_grad = grad/grad_norm.unsqueeze(-1)
        coords_3d = coords_3d - sdf*unit_grad
      
        coords = coords_3d.clone().cpu().detach()
        return {
            "coords": coords.float()
        }, {
            "normals": samples[:, 3:6].float(),
            "distances": samples[:, 6].float()
        }





class PointCloudImplictFunctions(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()
        self.coords = np.random.uniform(-1, 1, size=(300000, 3))
        self.points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        rand_idcs = np.random.choice(point_cloud_size, size=self.points)

        coords = torch.from_numpy(self.coords[rand_idcs, :]).float().unsqueeze(0)

        #function = implicit_functions.elipsoid()
        #function = implicit_functions.double_torus()
        function = implicit_functions.sdf_torus()
        function.eval()
        coord_values = function(coords)

        coords = coord_values['model_in']
        values = coord_values['model_out']#.unsqueeze(0)

        gradient = diff_operators.gradient(values, coords)#np.ones((off_surface_samples, 3)) * -1

        hessian = diff_operators.hessian(values, coords)
        min_curvature, max_curvature = diff_operators.principal_curvature(values, coords, gradient, hessian)
        principal_directions = diff_operators.principal_directions(gradient, hessian[0])[0]

        return {'coords': coords[0]}, {'sdf': values[0].cpu(),
                                       'normals': gradient[0].cpu(),
                                       'min_curvature': min_curvature.cpu(),
                                       'max_curvature': max_curvature.cpu(),
                                       'principal_directions': principal_directions[0].cpu()}



class reacDiffusion_sampler(Dataset):
    def __init__(self, training_steps = 10000, batch_size=0, silent=False):
        super().__init__()
        self.batch_size = batch_size

        self.elapsed_steps = 0
        self.training_steps = training_steps

    def __len__(self):
        lenght = 1
        return lenght

    def __getitem__(self, idx):
        return self._random_sampling(self.batch_size)
        #return self._random_sampling_with_tubular_neighborhood(self.batch_size)
        
    
    def add_approximated_values(self, samples_count=10, N=256):
        voxel_origin = [-1, -1]
        voxel_size = 2.0 /(N-1)

        overall_index = torch.arange(0, N ** 2, 1, out=torch.LongTensor())
        samples = torch.ones(N ** 2, 2)

        # transform first 2 columns
        # to be the x, y index
        samples[:, 1] = overall_index % N
        samples[:, 0] = (overall_index.long() / N) % N

        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[1]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[0]

        samples = samples.reshape((N,N,2))

        #u1 = torch.exp(-((samples[..., 0]**2)/0.5 + (samples[..., 1]**2)/0.5)).clone().detach()#torch.ones_like(samples[..., 0])
        u1 = torch.ones_like(samples[..., 0])
        u2 = torch.exp(-((samples[..., 0]**2)/0.01 + (samples[..., 1]**2)/0.01)).clone().detach()
        
        iterations = 1000
        delta_t = 1.0/iterations
        
        feed = 0.055
        kill = 0.062
        D1 = 8.0e-06
        D2 = D1/2

        time = 0.0

        xyt_sample = []
        u1_sample = []
        u2_sample = []

        for i in range(iterations):
            F = - u1*u2*u2 + feed*(1. - u1)
            G =   u1*u2*u2 - (feed + kill)*u2

            u1 += 100.0*(F + D1*diff_operators.discrete_laplacian(u1, delta=voxel_size))*delta_t
            u2 += 100.0*(G + D2*diff_operators.discrete_laplacian(u2, delta=voxel_size))*delta_t

            time += delta_t
            if (i%99) == 0 and i!=0:
                rand_idx = np.random.choice(N, (samples_count,2))
                xy_sample = samples[rand_idx[:, 0], rand_idx[:, 1], :].clone()
                xyt = torch.cat((xy_sample, time*torch.ones_like(xy_sample[...,0].unsqueeze(-1))), dim=-1)
                
                # plt.imshow(u2)
                # plt.show()

                if not len(xyt_sample):
                    u1_sample = u1[rand_idx[:, 0], rand_idx[:, 1]].clone()
                    u2_sample = u2[rand_idx[:, 0], rand_idx[:, 1]].clone()
                    
                    xyt_sample =  xyt              
                else:
                    u1_new = u1[rand_idx[:, 0], rand_idx[:, 1]].clone()
                    u1_sample = torch.cat((u1_sample, u1_new), dim=0 )
                    
                    u2_new = u2[rand_idx[:, 0], rand_idx[:, 1]].clone()
                    u2_sample = torch.cat((u2_sample, u2_new), dim=0 )
                
                    xyt_sample = torch.cat((xyt_sample, xyt), dim=0 )
        
        return xyt_sample, u1_sample, u2_sample
    
    
    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        
        on_zero_count = self.batch_size // 2
        off_zero_count =  self.batch_size+self.batch_size//2
        
        coords_at_zero  = torch.from_numpy(np.random.uniform(-1, 1, size=(on_zero_count, 2)).astype(np.float32))
        coords_off_zero = torch.from_numpy(np.random.uniform(-1, 1, size=(off_zero_count,2)).astype(np.float32))
        
        #samples_on_time = torch.from_numpy(np.random.uniform(0, 1.0*np.sqrt(self.elapsed_steps/self.training_steps), size=(off_zero_count,1)).astype(np.float32))
        samples_on_time = torch.from_numpy(np.random.uniform(0, 0.5, size=(off_zero_count,1)).astype(np.float32))
        
        off_time_zero = torch.cat((coords_off_zero, samples_on_time), dim=-1)
        on_time_zero = torch.cat((coords_at_zero, torch.zeros(on_zero_count, 1)), dim=-1)

        u1_0 = torch.ones(on_zero_count, 1)
        x1 = coords_at_zero[...,0].unsqueeze(-1)
        x2 = coords_at_zero[...,1].unsqueeze(-1)
        # u1_0 = torch.exp(-((x1**2)/0.5 + (x2**2)/0.5))
        # u1_0 = u1_0.clone().detach()
        u2_0 = torch.exp(-((x1**2)/0.01 + (x2**2)/0.01))
        u2_0 = u2_0.clone().detach()

        # Add approximated values
        add_approximated_values = False
        if add_approximated_values:
            xyt_sample, u1_sample, u2_sample = self.add_approximated_values(samples_count=1000)

            samples = torch.cat((on_time_zero, xyt_sample.detach(), off_time_zero), dim=0 )
            
            u_0 = torch.cat((u1_0, u2_0), dim=-1).detach()
            u_sample = torch.cat((u1_sample.detach().unsqueeze(-1), u2_sample.detach().unsqueeze(-1)), dim=-1).detach()

            function = torch.cat((u_0, u_sample,-1000*torch.ones(off_zero_count,2)), dim=0 )
        else:
            samples = torch.cat((on_time_zero, off_time_zero), dim=0 )
            
            u_0 = torch.cat((u1_0, u2_0), dim=-1).detach()
            
            function = torch.cat((u_0,-1000*torch.ones(off_zero_count,2)), dim=0 )            

        self.elapsed_steps += 1

        return {
            "coords": samples.detach().float()
        },{"function": function.detach().float()}


    
