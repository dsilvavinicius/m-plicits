import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from pytorch3d.io import load_ply, load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
import open3d as o3d

def load_input(file_path, device, num_points=10000):
    """
    Loads input data from a file, which can be either:
      - A mesh file (.obj, .ply) or 
      - A point cloud in .xyz format.
    
    For .xyz files, if there are extra columns (e.g., normals), only the first three columns are used.
    
    For .ply files, if the faces tensor is empty, the file is treated as a point cloud.
    
    Returns:
        A Tensor of shape (1, P, 3) representing the point cloud.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".xyz":
        # Load point cloud from an ASCII file (whitespace-separated).
        data = np.loadtxt(file_path)
        # If more than 3 columns, keep only the first three.
        if data.ndim == 2 and data.shape[1] >= 3:
            points_np = data[:, :3]
        else:
            points_np = data
        points = torch.tensor(points_np, dtype=torch.float32, device=device)
        # Ensure shape is (1, N, 3)
        if points.ndim == 2:
            points = points.unsqueeze(0)
        return points
    elif ext == ".ply":
        verts, faces = load_ply(file_path)
        verts = verts.to(device)
        faces = faces.to(device)
        # If faces is empty, treat the file as a point cloud.
        if faces.numel() == 0:
            points = verts.unsqueeze(0)  # Ensure shape is (1, N, 3)
            return points
        else:
            mesh = Meshes(verts=[verts], faces=[faces])
            points = sample_points_from_meshes(mesh, num_samples=num_points)
            return points
    elif ext == ".obj":
        mesh = load_objs_as_meshes([file_path], device=device)
        points = sample_points_from_meshes(mesh, num_samples=num_points)
        return points
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def voxelize_mesh_open3d(mesh_path, voxel_size=0.01):
    ext = os.path.splitext(mesh_path)[1].lower()

    if ext not in [".obj", ".ply"]:
        return None
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)

    # Extract voxel coordinates (integer grid indices)
    voxels = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    return voxels

def iou_from_voxels(vox1, vox2):
    set1 = set(map(tuple, vox1))
    set2 = set(map(tuple, vox2))

    inter = len(set1 & set2)
    union = len(set1 | set2)

    return inter / union if union > 0 else float("nan")

def mesh_iou(mesh1_path, mesh2_path, voxel_size=0.01):
    vox1 = voxelize_mesh_open3d(mesh1_path, voxel_size)
    vox2 = voxelize_mesh_open3d(mesh2_path, voxel_size)

    if vox1 is None or vox2 is None:
        return float("nan")
    
    return iou_from_voxels(vox1, vox2)

def compute_metrics(points1, points2, mesh1, mesh2, norm=2, iou_voxel_size=0.01):
    """
    Computes the Chamfer, Hausdorff and volumetric IoU (if meshes are available) between two shapes.
    Chamfer is computed with mean reduction and Hausdorff with max reduction.
    
    Returns:
        chamfer_loss, hausdorff_loss, iou.
    """
    chamfer_loss, _ = chamfer_distance(points1, points2, norm=norm)
    hausdorff_loss, _ = chamfer_distance(points1, points2, norm=norm, point_reduction="max")

    iou = mesh_iou(mesh1, mesh2, voxel_size=iou_voxel_size)
    
    return chamfer_loss.item(), hausdorff_loss.item(), iou

def main(json_file, csv_file, latex_file, num_points=10000, norm=2, iou_voxel_size=0.01):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Read the JSON file containing pairs of file paths.
    with open(json_file, 'r') as f:
        file_pairs = json.load(f)
    
    results = []
    for pair in file_pairs:
        file1 = pair["mesh1"]
        file2 = pair["mesh2"]
        
        # Load each file (mesh or point cloud) as a point cloud.
        points1 = load_input(file1, device, num_points=num_points)
        points2 = load_input(file2, device, num_points=num_points)
        
        # Compute the metrics.
        chamfer_val, hausdorff_val, iou = compute_metrics(points1, points2, file1, file2, norm=norm, iou_voxel_size=iou_voxel_size)

        results.append({
            "File1": os.path.basename(file1),
            "File2": os.path.basename(file2),
            "Chamfer": chamfer_val,
            "Hausdorff": hausdorff_val,
            "IoU": iou
        })
    
    # Save results to CSV.
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    
    # Save results as a LaTeX table.
    latex_table = df.to_latex(index=False, float_format="%.6f")
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    
    print("Results saved:")
    print(" CSV:", csv_file)
    print(" LaTeX table:", latex_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Chamfer, Hausdorff, and volumetric IoU (if inputs are meshes) between pairs of meshes or point clouds."
    )
    parser.add_argument("--json_file", type=str, default="mesh_pairs.json",
                        help="Path to JSON file with pairs. Each pair should have keys 'mesh1' and 'mesh2'.")
    parser.add_argument("--csv_file", type=str, default="results.csv", help="Output CSV file.")
    parser.add_argument("--latex_file", type=str, default="results_table.tex", help="Output LaTeX file with table.")
    parser.add_argument("--num_points", type=int, default=10000, help="Number of points to sample for meshes.")
    parser.add_argument("--norm", type=int, default=2, choices=[1, 2])
    parser.add_argument("--iou_voxel_size", type=float, default=0.01, choices=[0.005, 0.01, 0.02])
    args = parser.parse_args()
    main(args.json_file, args.csv_file, args.latex_file, args.num_points, args.norm, args.iou_voxel_size)