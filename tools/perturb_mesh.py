#!/usr/bin/env python3
import argparse
import numpy as np
import open3d as o3d
import os

def perturb_and_save(input_path: str, output_ply: str, delta_frac: float):
    """
    Load a mesh (OBJ or PLY), compute its bounding‐box diagonal,
    then jitter each vertex along its normal by up to ±(delta_frac × diagonal),
    and save the perturbed mesh as a PLY.
    """
    # 1) Read mesh
    mesh = o3d.io.read_triangle_mesh(input_path,
                                     enable_post_processing=True)

    # 2) Ensure vertex normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # 3) Compute bounding‐box diagonal length
    bbox   = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())            # [dx, dy, dz]
    diag   = np.linalg.norm(extent)                   # diagonal length

    # 4) Convert delta_frac → absolute delta
    abs_delta = delta_frac * diag

    print(f"Perturbing vertices by ±{abs_delta:.4f} units (±{delta_frac * 100:.2f}% of bounding box diagonal)")

    # 5) Grab vertices & normals as NumPy
    verts   = np.asarray(mesh.vertices)               # (V,3)
    normals = np.asarray(mesh.vertex_normals)         # (V,3)

    # 6) Sample uniform noise in [–abs_delta, +abs_delta] per vertex
    noise = np.random.uniform(-abs_delta, abs_delta, size=(verts.shape[0], 1))

    # 7) Perturb positions
    verts_perturbed = verts + normals * noise

    # 8) Overwrite and export
    mesh.vertices = o3d.utility.Vector3dVector(verts_perturbed)
    o3d.io.write_triangle_mesh(output_ply, mesh, write_ascii=False)

    base, _      = os.path.splitext(output_ply)  
    output_obj   = base + ".obj"

    # 3) Write the same mesh to OBJ
    o3d.io.write_triangle_mesh(output_obj, mesh, write_ascii=False)  

    print("Perturbed mesh saved to:", output_ply)
    print(".obj also saved in :", output_obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perturb mesh vertices along normals (delta as % of bbox diag) and save to PLY"
    )
    parser.add_argument("--input",  help="Path to the .obj or .ply input mesh")
    parser.add_argument("--output", help="Path to save the perturbed .ply mesh")
    parser.add_argument(
        "--delta", type=float, default=0.01,
        help="Fraction of bounding‐box diagonal to use as max perturbation (e.g. 0.01 = 1%)"
    )
    args = parser.parse_args()
    perturb_and_save(args.input, args.output, args.delta)
