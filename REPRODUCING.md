# Reproducing the paper's experiments

Everything runs from the repository root with the data archive unpacked in
place (`python tools/download_data.py`) and the training environment active
(`conda env create -f environment.yml && conda activate new_i3d && pip install -e .`).
The evaluation metrics need PyTorch3D and run in the `metrics` environment
(`metrics/environment.yml`); `reproduce.py` finds it next to the active one
or through `MPLICITS_METRICS_PYTHON`.

**Hardware note.** The main-paper timings (Tab. 2 training/sampling times,
Tab. 4/5 FPS and extraction) were measured on a single NVIDIA RTX 5090; the
appendix experiments (S2.3–S2.4, S3.1–S3.4) ran on a single NVIDIA L40S.
Quality metrics are hardware-independent; timing columns are not.

## 0. One command

```bash
python reproduce.py all            # Tab. 2 ours rows, extraction times, Tab. 4 -> reproduction/REPORT.md
python reproduce.py paper-noise    # Tab. 3 ours rows from the paper's reconstructions (optional archive)
python reproduce.py train --shapes normalized_armadillo_gt    # retrain from scratch and compare with the release
python reproduce.py noise --shapes normalized_armadillo_gt 354371   # Tab. 3 protocol on a subset
```

The report lists every reproduced number next to the published one. The
sections below map each table to the underlying commands, for anyone who
wants to run a piece by hand.

## 1. Training the released models (Tab. 2 "ours" rows)

Per shape, three stages (coarse → medium residual → fine residual). The
configuration that trained each released checkpoint ships next to it as
`results/<shape>/<stage>/config.yaml`, and the trainers read that file
directly, so retraining is:

```bash
S=normalized_armadillo_gt
python train_sdf.py data/normalized_sphere/input/$S.ply out/$S/coarse results/$S/coarse/config.yaml
python experiment_scripts/train_sdf_on_neighborhood.py data/normalized_sphere/input/$S.ply out/$S/medium results/$S/medium/config.yaml out/$S/coarse/best.pth
python experiment_scripts/train_sdf_on_neighborhood_fine.py data/normalized_sphere/input/$S.ply out/$S/fine results/$S/fine/config.yaml out/$S/coarse/best.pth out/$S/medium/best.pth
```

(`reproduce.py train` does exactly this and evaluates the result.) The
`experiments/*.yaml` files are older exploratory configs used by the
appendix ablations; they are not the released models. Checkpoints are saved
with the SIREN frequency baked into the weights, which is why every
downstream tool loads them with `w0 = 1`.

## 2. Reconstruction (marching cubes; Tab. 2 sampling column, Tab. 5)

```bash
# multiscale, band-culled (the paper's inference)
python reconstruct.py results/$S/fine/best.pth out.ply -r 512 --device cuda --multistage \
    --coarse_path results/$S/coarse/best.pth --medium_path results/$S/medium/best.pth
# the same model evaluated everywhere (no culling): the Tab. 5 baseline analogue
python reconstruct.py results/$S/fine/best.pth out_unculled.ply -r 512 --device cuda \
    --coarse_path results/$S/coarse/best.pth --medium_path results/$S/medium/best.pth
```

`reconstruct.py` prints `SAMPLING_TIME_S:<seconds>`; `reproduce.py recon`
collects it for every shape. Tab. 5's exact rows use a separate
(64,1)▷(128,2)▷(400,2) family and a (256,4) SIREN baseline that are not
part of the release; the culled-vs-unculled timing of the released models
reproduces the effect, not the exact numbers.

## 3. Evaluation protocol (CD / Hausdorff / IoU; Tabs. 2, 3)

The paper's metric implementation is `metrics/meshes/compute_distance_metrics.py`
(500K samples, squared-L2 bidirectional Chamfer, voxel IoU at 0.01):

```bash
python metrics/meshes/compute_distance_metrics.py --json_file pairs.json --csv_file out.csv \
    --latex_file out.tex --num_points 500000 --norm 2 --iou_voxel_size 0.01
```

where `pairs.json` lists `{"mesh1": reconstruction, "mesh2": ground truth}`
pairs (ground truth = the clean input cloud in `data/normalized_sphere/input/`).
`reproduce.py metrics` builds the pairs and aggregates mean/median per level.
Baseline glue code (iNGP / BACON / IDF harnesses) is in
`metrics/comparison_for_mplicits/`; the baselines themselves come from their
official repositories.

Tab. 3 (1% vertex noise) trains on `data/normalized_noise/noise_data/<shape>.ply`
with the same configs and evaluates against the clean ground truth:
`reproduce.py noise --shapes ...` (about 8 minutes per shape on an RTX 5090;
36 shapes for the full table). The paper's own noise reconstructions ship
in the optional `m-plicits-paper-meshes.zip`, so `reproduce.py paper-noise`
recomputes the table's ours rows without retraining.

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

Build and usage in `renderer/README.md` (Windows + CUDA). The registry's
`armadillo` and `thai_statue` entries are the released checkpoints byte for
byte; any other released model loads at runtime through
`renderer/scripts/export_experiment.py` + `-experiment_file`. Tab. 4 is the
Armadillo at 512², 20 sphere-tracing iterations on the first SDF and 5 per
subsequent level; `-normal_lod` selects the level whose analytic normals
shade the hit (neural normal mapping when it exceeds the last traced level).
`reproduce.py fps` runs the rows below with 500-frame benchmarks:

```bash
cd renderer/cuda/build/Release
MIP-plicitsRenderer.exe -experiment=armadillo_siren -benchmark=500                                              # (400,4) SIREN baseline
MIP-plicitsRenderer.exe -experiment=armadillo -benchmark=500                                                    # (128,2) coarse
MIP-plicitsRenderer.exe -experiment=armadillo -benchmark=500 -iters=20,0,0 -delta=0.02 -normal_lod=1            # (128,2)>(256,2) (NM)
MIP-plicitsRenderer.exe -experiment=armadillo -benchmark=500 -iters=20,5,0 -delta=0.02 -normal_lod=1            # (128,2)>(256,2)
MIP-plicitsRenderer.exe -experiment=armadillo -benchmark=500 -iters=20,5,0 -delta=0.02 -normal_lod=2            # full (NM)
MIP-plicitsRenderer.exe -experiment=armadillo -benchmark=500 -iters=20,5,5 -delta=0.02 -normal_lod=2            # full
```

The two-level (128,2)▷(400,2) rows use a separately trained model that is
not part of the release. Benchmark mode redraws continuously; the
interactive window is capped near 100 FPS by its refresh timer.

## 6. Attribute (texture) mapping (suppl. attribute section)

Texture training/evaluation in `attributes/README.md` (its own conda env;
PyTorch3D). Reproduces the texture figures and the texture MSE table; the
"Egg" row's asset is not part of the released archive. Neural *normal*
mapping needs no training: it is the analytic gradient of a finer SDF,
rendered by the real-time renderer (`-normal_lod`, NM rows of Tab. 4).

## 7. Verification log

**2026-09-16, fresh clone, Windows 11, RTX 5090, CUDA 12.8.** Repository
cloned into an empty folder, archive unpacked, environments created from
the shipped files, renderer built from scratch, then `reproduce.py` for
every stage. Full per-shape output in the run's `reproduction/` folder;
the aggregates:

### Tab. 2, ours rows (released checkpoints, clean inputs, 512³, 500K samples)

| level | shapes | mean CD | median CD | mean IoU | median IoU |
|---|---|---|---|---|---|
| coarse | 36 | 3.90E-05 | 2.17E-05 | 0.522 | 0.579 |
| coarse, paper as printed | 37 | 5.36E-05 | 3.80E-05 | 0.448 | 0.459 |
| fine, Eq. 5 adaptive bands (`reconstruct.py --multistage --input`) | 34 | 2.01E-05 | 1.14E-05 | 0.723 | 0.869 |
| fine, paper as printed | 37 | 6.57E-05 | 1.87E-05 | 0.586 | 0.867 |
| fine, fixed bands 0.1/0.06 (`--deltas 0.1 0.06`) | 34 | 1.40E-04 | 6.87E-05 | 0.489 | 0.583 |
| fine, unculled (all levels everywhere) | 34 | 1.95E-04 | 7.49E-05 | 0.466 | 0.469 |

Two facts about provenance, so the comparison is read correctly:

- The released checkpoints reproduce the paper's own March mesh
  generation per shape: recomputing that generation with the same metric
  code gives coarse 3.90E-05 / 2.16E-05 / 0.522 / 0.579 and fine
  1.42E-04 / 7.96E-05 / 0.489 / 0.583; the rows above match them
  (median CD ratio 1.00, 68 of 69 shape-levels within 25%). Those March
  fine meshes were extracted with the fixed 0.1/0.06 band widths that
  `reconstruct.py` carried until this release.
- The printed Tab. 2 ours rows were produced by a collaborator's
  reconstruction pass whose meshes are not on the machine used here, so
  they cannot be re-derived exactly; the reproducible numbers with the
  paper's described inference (adaptive bands) are at or above the
  printed fine row on every column and above the printed coarse row.
  The shape counts differ because 58168 has no fine checkpoint and the
  Thai statue only a coarse one.

Degenerate shapes worth knowing about when reading means: coarse 95444
(CD 2.4E-04) and the Thai statue (2.5E-04); fine 95444 (1.7E-04). The
Armadillo's fine level carries a small detached component away from the
surface (IoU 0.05, CD 5.3E-05).

### Extraction time at 512³ (Tab. 2 sampling column; Tab. 5 analogue)

| extraction | shapes | mean seconds | paper |
|---|---|---|---|
| coarse only | 36 | 1.26 | 1.51 |
| multiscale, band-culled | 36 | 3.07 | 5.29 |
| multiscale, unculled | 36 | 6.14 | Tab. 5 baseline analogue (2.0× slower) |

### Tab. 3, ours rows (the paper's noise reconstructions, `reproduce.py paper-noise`)

35 shapes; 58168 excluded (its stored input was repaired after those
meshes were made, see PROVENANCE).

| level | mean CD | median CD | mean IoU | median IoU | paper (mean / median CD, mean / median IoU) |
|---|---|---|---|---|---|
| coarse | 5.80E-05 | 4.43E-05 | 0.442 | 0.467 | 7.42E-05 / 5.33E-05, 0.381 / 0.403 |
| medium | 3.84E-05 | 3.03E-05 | 0.551 | 0.634 | 4.62E-05 / 3.03E-05, 0.537 / 0.638 |
| fine | 2.59E-05 | 1.65E-05 | 0.529 | 0.548 | 3.36E-05 / 1.62E-05, 0.582 / 0.591 |

Retraining on the noisy input from scratch (`reproduce.py noise`), fine
level against the clean ground truth: Armadillo CD 1.38E-05 / IoU 0.869,
Thingi 354371 CD 1.20E-05 / IoU 0.859.

### Tab. 4, real-time renderer (Armadillo, 512², 500 frames)

| row | reproduced FPS | paper FPS |
|---|---|---|
| (400,4) SIREN baseline | 16 | 19 |
| (128,2) coarse | 118 | 180 |
| (128,2)▷(256,2) (NM) | 64 | 85 |
| (128,2)▷(256,2) | 62 | 70 |
| (128,2)▷(256,2)▷(400,2) (NM) | 35 | 43 |
| (128,2)▷(256,2)▷(400,2) | 31 | 35 |

Same ordering and ratios; the absolute values are 65–90% of the printed
ones on this build (sm_86 binary run through the driver's JIT on an
sm_120 GPU gives the same numbers as a native sm_120 build).

### Training from scratch (`reproduce.py train`, Armadillo, shipped configs)

The retrained coarse level is identical to the released one (CD 2.07E-05,
IoU 0.778, fixed seed); the retrained fine level scores CD 4.89E-05
against 5.26E-05 for the released checkpoint. Wall time 2.5 + 1.0 + 1.9
minutes for the three stages (paper averages: 4.07 coarse, 8.53 full).

### Texture mapping

`attributes/`: 100-epoch texture training on `spot` and colored-mesh
extraction run end to end in the `neural_textures` environment.
