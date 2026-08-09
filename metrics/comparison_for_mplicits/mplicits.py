import argparse
import subprocess
import sys
import os


def _setup_project_imports():
    """Add the project root to sys.path so sibling modules can be imported."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(
        description="Activate a conda environment, run an mplicits script, then generate mesh comparisons and compute distance metrics."
    )
    parser.add_argument(
        "conda_env",
        help="Name of the conda environment to activate.",
    )
    parser.add_argument(
        "script_path",
        help="Absolute path of the Python script to run.",
    )

    # ----- mplicits parameters -----
    parser.add_argument("--input", required=True, help="Input directory path (ground-truth meshes).")
    parser.add_argument("--ground_truth", required=False, help="Folder with ground truth meshes for comparison. If missing, --input is used.")
    parser.add_argument("--output", required=True, help="Output directory path (reconstructed meshes and results).")
    parser.add_argument("--device", default=None, help="Device to use (e.g. cuda:0).")
    parser.add_argument("--grid_resolution", type=int, default=None, help="Grid resolution for reconstruction.")

    # ----- Metrics parameters -----
    parser.add_argument("--num_points", type=int, default=10000,
                        help="Number of points to sample from each mesh (default: 10000).")
    parser.add_argument("--norm", type=int, default=2, choices=[1, 2],
                        help="Norm to use for Chamfer/Hausdorff (default: 2).")
    parser.add_argument("--iou_voxel_size", type=float, default=0.01, choices=[0.005, 0.01, 0.02],
                        help="Voxel size for IoU computation (default: 0.01).")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Run the mplicits reconstruction inside the specified conda env
    # ------------------------------------------------------------------
    command = [
        #"conda",
        #"run",
        #"-n",
        #args.conda_env,
        #"--no-capture-output",
        "C:/Program Files/Git/bin/bash.exe",
        args.script_path,
        args.input,
    ]

    # Optional positional arguments
    if args.output:
        command.append(args.output)
    if args.device:
        command.append(args.device)
    if args.grid_resolution is not None:
        command.append(str(args.grid_resolution))

    print(f"Activating conda environment '{args.conda_env}' and running script...")
    print(f"Command: {' '.join(command)}\n")

    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"\nERROR: mplicits script exited with return code {result.returncode}")
        sys.exit(result.returncode)

    # ------------------------------------------------------------------
    # 2. Generate mesh-comparison JSON (input <-> reconstruction pairs)
    # ------------------------------------------------------------------
    _setup_project_imports()
    from generate_mesh_comparisons import generate_pairs
    from meshes.compute_distance_metrics import main as compute_metrics_main

    json_file = os.path.join(args.output, "mesh_comparisons.json")

    ground_truth_folder = args.ground_truth if args.ground_truth else args.input

    print("\n--- Generating mesh comparison pairs ---")
    generate_pairs(
        input_folder=ground_truth_folder,
        reconstructions_folder=args.output,
        output=json_file,
        suffixes=["_coarse", "_medium", "_fine"]
    )

    # DEBUG
    return 

    # ------------------------------------------------------------------
    # 3. Compute distance metrics (Chamfer, Hausdorff, IoU)
    # ------------------------------------------------------------------
    csv_file = os.path.join(args.output, "results.csv")
    latex_file = os.path.join(args.output, "results_table.tex")

    print("\n--- Computing distance metrics ---")
    compute_metrics_main(
        json_file=json_file,
        csv_file=csv_file,
        latex_file=latex_file,
        num_points=args.num_points,
        norm=args.norm,
        iou_voxel_size=args.iou_voxel_size,
    )

    print("\nDone. All outputs are in:", args.output)


if __name__ == "__main__":
    main()
