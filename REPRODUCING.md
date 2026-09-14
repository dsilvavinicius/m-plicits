# Reproducing the paper's experiments

Every table and figure of the paper maps to a command below. Commands are
given from the repository root, with the data archive unzipped in place
(see `data/README.md`) and the training environment active
(`conda activate new_i3d` after `conda env create -f environment.yml`;
`pip install -e .`).

**Hardware note.** The main-paper timings (Tab. 2 training/sampling times,
Tab. 4/5 FPS and extraction) were measured on a single NVIDIA RTX 5090; the
appendix experiments (S2.3–S2.4, S3.1–S3.4) ran on a single NVIDIA L40S.
Quality metrics are hardware-independent; timing columns are not.

## 1. Training the released models (Tab. 2 "ours" rows)

Per shape, three stages (coarse → medium residual → fine residual), using
the per-shape configs in `experiments/`. Config naming varies by shape; the
authoritative configuration for every released checkpoint is stored next to
it as `results/<shape>/<stage>/config.yaml` in the data archive:

```bash
python train_sdf.py data/normalized_sphere/input/<shape>.ply results/<shape>/coarse experiments/<shape>_coarse.yaml
python experiment_scripts/train_sdf_on_neighborhood.py data/normalized_sphere/input/<shape>.ply results/<shape>/medium experiments/<shape>_on_neigh_medium.yaml results/<shape>/coarse/best.pth
python experiment_scripts/train_sdf_on_neighborhood_fine.py data/normalized_sphere/input/<shape>.ply results/<shape>/fine experiments/<shape>_on_neigh_fine.yaml results/<shape>/coarse/best.pth results/<shape>/medium/best.pth
```

The data archive already ships trained checkpoints in `results/`, so every
downstream step (reconstruction, rendering, metrics, attribute mapping)
also works without retraining. `experiment_scripts/train_sdf_end2end.py`
automates the three stages and the noise sweep used by Tab. 3.

## 2. Reconstruction (marching cubes, Tab. 5)

```bash
# multiscale, band-culled (the paper's inference; Tab. 5 "multiscale")
python reconstruct.py results/<shape>/fine/best.pth out.ply -r 512 --multistage \
    --coarse_path results/<shape>/coarse/best.pth --medium_path results/<shape>/medium/best.pth
```

## 3. Evaluation protocol (CD / Hausdorff / IoU — Tabs. 2, 3)

The paper's metric implementation lives in
`metrics/meshes/compute_distance_metrics.py` (500K samples, squared-L2
bidirectional Chamfer, voxel-shell IoU at 0.01, run inside the `metrics`
environment — `metrics/environment.yml`, PyTorch3D required). The wrapper

```bash
python appendix_experiments/run_metrics.py --meshes-dir <dir-of-meshes> --out-prefix <prefix>
```

pairs reconstructions with their ground truth in
`data/normalized_sphere/input/` and produces a CSV. Baseline glue code
(iNGP / BACON / IDF harnesses) is in `metrics/comparison_for_mplicits/`;
the baselines themselves come from their official repositories.

## 4. Appendix experiments

All drivers live in `appendix_experiments/` and write to subfolders there;
`--smoke` runs a minutes-long plumbing check of the full chain.

| Paper section | Command |
|---|---|
| S3.1 isolation ablation (cells a/b/c) | `python appendix_experiments/run_ablation.py` then `run_metrics.py` + `summarize.py`. Add `--fresh-a` to retrain cell (a) on clean inputs instead of reusing the released checkpoints. Cell (d) (matched-parameter monolith) is also implemented (`--cells d`). |
| S3.1 unculled re-extraction | `python appendix_experiments/run_nocull.py` (+ `recompute_struct.py`, `nocull_faces.py` for components/CD of meshes beyond PyTorch3D's sampler limit) |
| S3.1 component census | `python appendix_experiments/blob_census.py` |
| S2.3 Screened Poisson | `python appendix_experiments/run_spsr.py` (trimmed) and `run_spsr_notrim.py` |
| S2.4 BANF re-implementation | `python appendix_experiments/banf/run_banf_probe.py --shape <shape> --cond clean|noise` (per-level fidelity probe + final meshes); component counts via `banf_comps.py` |
| S3.2 coarse under-training | `python appendix_experiments/run_undertrained.py` |
| S3.3 outlier contamination | `python appendix_experiments/run_outliers.py`; band-width verification via `recheck_outlier_deltas.py` |
| S3.4 measured band widths | `python appendix_experiments/extract_deltas.py` |
| S3 loss/δ ablations (Tabs. S8–S11) | `experiment_scripts/ablation_loss.sh`, `ablation_delta.py`, `ablation_delta_fine_level.py` |

Noise inputs for Tab. 3 / the appendix are shipped in the archive; to
regenerate them from scratch use `tools/perturb_mesh.py` (general) or
`appendix_experiments/make_fixed_noise.py` (the exact seeded appendix
inputs). `appendix_experiments/fix_thai_normals.py` documents the
zero-length-normal repair applied to the Thai input.

## 5. Real-time renderer (Tab. 4; FPS columns of Tab. 2)

Build and usage in `renderer/README.md` (Windows + CUDA). The new
`-benchmark=<N>` mode prints and logs the average FPS, replacing the
window-title readout used for the paper. Tab. 4's protocol is 512²
resolution (default build) with 20 sphere-tracing iterations on the first
SDF and 5 per subsequent level; the per-row configuration is passed on the
command line (the defaults are coarse-only). Absolute FPS is
hardware-dependent (the paper used an RTX 5090); relative orderings between
rows are the reproducible claim.

```bash
renderer/cuda/build/Release/MIP-plicitsRenderer.exe -list
# Tab. 4 rows: coarse only / full detail / neural normal mapping
renderer/cuda/build/Release/MIP-plicitsRenderer.exe -experiment=armadillo -benchmark=500
renderer/cuda/build/Release/MIP-plicitsRenderer.exe -experiment=armadillo -benchmark=500 -iters=20,5,5 -delta=0.02 -normal_lod=2
renderer/cuda/build/Release/MIP-plicitsRenderer.exe -experiment=armadillo -benchmark=500 -iters=20,5,0 -delta=0.02 -normal_lod=2
```

## 6. Attribute (texture) mapping — suppl. attribute section

Texture training/evaluation in `attributes/README.md` (its own conda env;
PyTorch3D). Reproduces the texture figures and the texture MSE table; the
"Egg" row's asset is not part of the released archive. Neural *normal*
mapping needs no training — it is the analytic gradient of a finer SDF,
rendered by the real-time renderer (normal-mapping toggle / NM rows of
Tab. 4).
