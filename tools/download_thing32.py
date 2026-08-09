#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import requests
from pathlib import Path
import numpy as np
from typing import List, Optional
import open3d as o3d


def download_thingi10k_files(
        model_ids: List[str], output_folder: str, skip_if_exists: Optional[bool] = True,
        scale_to_unity: Optional[bool] = True
) -> List[str]:
    """Download files from the Thingi10k dataset.

    Parameters
    ----------
    model_ids: List[str]
        List of model IDs to download.

    output_folder: str, PathLike
        Path to the output folder where files will be saved. Will be created if
        it doesn't exist.

    skip_if_exists: bool, optional
        Default is True. Skip if the file already exists.



    Returns
    -------
    downloaded_files: List[Path]
        List with the output files.
    """
    # Ensure the output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Base URL for downloading Thingi10k models
    base_url = (
        "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main/raw_meshes/"
    )

    output_files = []
    for model_id in model_ids:
        output_file = os.path.join(output_folder, f"{model_id}.stl")
        output_files.append(output_file)
        if skip_if_exists and os.path.exists(output_file):
            print(f'File for "{model_id}" exists. Skipping.')
            continue

        try:
            response = requests.get(f"{base_url}{model_id}.stl")
            # Check if the request was successful
            if response.status_code == 200:
                with open(output_file, "wb") as f:
                    f.write(response.content)

                print(f"Successfully downloaded model {model_id} to {output_file}")
            else:
                print(
                    f"Failed to download model {model_id}: Status code {response.status_code}"
                )
        except Exception as e:
            print(f"Error downloading model {model_id}: {e}")
        else:
            if scale_to_unity:
                mesh = (
                    o3d.io.read_triangle_mesh(output_file)
                    .remove_duplicated_vertices()
                    .remove_degenerate_triangles()
                    .remove_non_manifold_edges()
                )
                center = mesh.get_center()
                mesh = mesh.translate(-center, relative=True)
                v = np.asarray(mesh.vertices)
                # The 1.1 below serves to leave a gap between the mesh and bbox border.
                max_distance = np.linalg.norm(v, axis=1).max() * 1.1
                mesh = mesh.scale(1.0 / max_distance, mesh.get_center())
                mesh.compute_triangle_normals(normalized=True)
                o3d.io.write_triangle_mesh(output_file, mesh)

    return output_files


def convert_stl_to_ply(
    files: List[str], output_folder: str, skip_if_exists: Optional[bool] = True
):
    """Convert STL files to PLY format.

    Parameters
    ----------
    files: List[str]
        List of paths to the STL files

    output_folder: str
        Path to the output folder where PLY files will be saved. Will be
        created if it doesn't exist.

    skip_if_exists: bool, Optional
        Skips processing if the output PLY already exists. Default is True.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each file in the input folder
    for filepath in files:
        ply_file = os.path.join(
            output_folder, os.path.split(filepath)[-1].replace(".stl", ".ply")
        )
        if skip_if_exists and os.path.exists(ply_file):
            print(f'File "{ply_file}" already exists. Skipping.')
            continue
        if filepath.endswith(".stl") and os.path.exists(filepath):
            print(f'Processing "{filepath}"')
            mesh = (
                o3d.io.read_triangle_mesh(filepath)
                .remove_duplicated_vertices()
                .remove_degenerate_triangles()
                .remove_non_manifold_edges()
            )
            o3d.io.write_triangle_mesh(ply_file, mesh)
            print(f"Successfully converted {filepath} to {ply_file}")


if __name__ == "__main__":
    # From https://github.com/nv-tlabs/nglod/issues/4
    model_ids = [
        "44234",
        "96481",
        "68381",
        "68380",
        "78671",
        "64444",
        "252119",
        "75665",
        "75656",
        "527631",
        "79241",
        "398259",
        "73075",
        "316358",
        "75496",
        "313444",
        "90889",
        "441708",
        "53159",
        "64764",
        "92763",
        "75662",
        "77245",
        "58168",
        "75655",
        "95444",
        "92880",
        "72870",
        "354371",
        "47984",
        "72960",
        "76277",
    ]

    parser = argparse.ArgumentParser(
        description="Downloads the thingi32 dataset. Optionally, converts it to PLY."
    )
    parser.add_argument(
        "outputpath",
        help="Path to the output folder (will be created if necessary)."
    )
    parser.add_argument(
        "--convert-to-ply", "-c", action="store_true",
        help="Converts the downloaded files to the Stanford PLY formate."
    )
    parser.add_argument(
        "--scale-to-unity-cube", "-s", action="store_true",
        help="Scales the output models to fit inside the unity cube [-1, 1]^3."
        " In practice the models will be scaled to [-0.9, 0.9]^3 to leave a"
        " small gap between the object and the boundary."
    )
    args = parser.parse_args()

    downloaded_files = download_thingi10k_files(
        model_ids, args.outputpath, skip_if_exists=True,
        scale_to_unity=args.scale_to_unity_cube
    )
    if args.convert_to_ply:
        convert_stl_to_ply(downloaded_files, args.outputpath)
