import argparse
import csv
import subprocess
import sys
import os
from pathlib import Path


def _setup_project_imports():
    """Add the project root to sys.path so sibling modules can be imported."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(
        description="Run IDF SDF training, then generate mesh comparisons and compute distance metrics."
    )
    parser.add_argument(
        "conda_env",
        help="Name of the conda environment to activate.",
    )
    parser.add_argument(
        "script_path",
        help="Absolute path of the .sh or .py training script.",
    )

    # ----- IDF parameters -----
    parser.add_argument("--input", required=True, help="Input directory path (ground-truth meshes).")
    parser.add_argument("--output", required=True, help="Output file path (reconstructed mesh).")
    parser.add_argument("--grid_resolution", type=int, default=None, help="Optional reconstruction grid size parameter.")

    # ----- Metrics parameters -----
    parser.add_argument("--num_points", type=int, default=10000,
                        help="Number of points to sample from each mesh (default: 10000).")
    parser.add_argument("--norm", type=int, default=2, choices=[1, 2],
                        help="Norm to use for Chamfer/Hausdorff (default: 2).")
    parser.add_argument("--iou_voxel_size", type=float, default=0.01, choices=[0.005, 0.01, 0.02],
                        help="Voxel size for IoU computation (default: 0.01).")
    parser.add_argument("--ground_truth", default=None,
                        help="Optional folder with ground-truth meshes for metrics. "
                             "If provided, used instead of --input for generate_pairs.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Train an SDF for every mesh in the input folder
    # ------------------------------------------------------------------
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    mesh_files = input_dir.glob("*.ply")

    if not mesh_files:
        print(f"No .ply files found in {input_dir}")
        sys.exit(1)

    for mesh_file in mesh_files:
        output_filename = mesh_file.name
        experiment_name = mesh_file.stem

        # Skip meshes that have already been processed (resume support)
        output_mesh = output_dir / output_filename
        stats_file = output_dir / f"{experiment_name}_stats.csv"
        if output_mesh.exists() and stats_file.exists():
            print(f"\n--- Skipping {mesh_file.name} (output mesh and stats already exist) ---")
            continue

        runner = "python" if str(args.script_path).endswith(".py") else "bash"
        command = [
            "conda", "run", "-n", args.conda_env, "--no-capture-output",
            runner, args.script_path, str(mesh_file), str(output_dir),
        ]
        if args.conda_env == "idf":
            command.extend([output_filename, experiment_name])
        if args.grid_resolution is not None:
            command.append(str(args.grid_resolution))

        print(f"\n--- Training SDF for {mesh_file.name} ---")
        print(f"Command: {' '.join(command)}\n")

        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"\nERROR: IDF training for {mesh_file.name} exited with return code {result.returncode}")
            sys.exit(result.returncode)

    # ------------------------------------------------------------------
    # 2. Generate mesh-comparison JSON (input <-> reconstruction pairs)
    # ------------------------------------------------------------------
    _setup_project_imports()
    from generate_mesh_comparisons import generate_pairs
    from meshes.compute_distance_metrics import main as compute_metrics_main

    json_file = os.path.join(output_dir, "idf_comparisons.json")

    gt_folder = args.ground_truth if args.ground_truth else args.input

    print("\n--- Generating mesh comparison pairs ---")
    generate_pairs(
        input_folder=gt_folder,
        reconstructions_folder=output_dir,
        output=json_file,
        suffixes=[""],
    )

    # ------------------------------------------------------------------
    # 3. Compute distance metrics (Chamfer, Hausdorff, IoU)
    # ------------------------------------------------------------------
    csv_file = os.path.join(output_dir, "results.csv")
    latex_file = os.path.join(output_dir, "results_table.tex")

    print("\n--- Computing distance metrics ---")
    compute_metrics_main(
        json_file=json_file,
        csv_file=csv_file,
        latex_file=latex_file,
        num_points=args.num_points,
        norm=args.norm,
        iou_voxel_size=args.iou_voxel_size,
    )

    # ------------------------------------------------------------------
    # 4. Compile per-mesh statistics into a single statistics.csv
    # ------------------------------------------------------------------
    stats_files = sorted(output_dir.glob("*_stats.csv"))
    if stats_files:
        statistics_csv = os.path.join(output_dir, "statistics.csv")
        print(f"\n--- Compiling {len(stats_files)} stats file(s) into statistics.csv ---")

        header = None
        rows = []
        for sf in stats_files:
            with open(sf, newline="") as f:
                reader = csv.reader(f)
                file_header = next(reader)
                if header is None:
                    header = file_header
                for row in reader:
                    rows.append(row)

        with open(statistics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"Wrote {statistics_csv} ({len(rows)} row(s))")
    else:
        print("\nNo *_stats.csv files found; skipping statistics.csv compilation.")

    print("\nDone. All outputs are in:", output_dir)


if __name__ == "__main__":
    main()
