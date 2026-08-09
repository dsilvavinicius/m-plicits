# M-plicits: Neural Implicit Surfaces via Nested Multiscale Residuals

Official code release. Project page: `docs/` (served via GitHub Pages once
enabled: Settings &rarr; Pages &rarr; deploy from `main` / `docs`). M-plicits models a signed distance function as a
base SIREN plus a sequence of residual SIRENs, each supervised only inside
the adaptive δ-band of the previous level's zero-level set. The nesting is
a training-time invariant that the inference algorithms exploit: multiscale
sphere tracing, band-culled marching cubes, GEMM-based analytical normals,
and neural normal/texture mapping — real-time rendering from a compact
model, with strong robustness to input noise.

## Repository map

| Folder | Contents |
|---|---|
| `i3d/`, `train_sdf.py`, `experiment_scripts/`, `reconstruct.py` | Training: coarse stage + banded residual stages, multiscale reconstruction |
| `experiments/` | Per-shape training configs |
| `appendix_experiments/` | Drivers for every appendix experiment (isolation ablation, SPSR, BANF re-implementation, robustness studies) |
| `metrics/` | The paper's evaluation protocol (CD / Hausdorff / voxel IoU) + baseline harnesses |
| `renderer/` | Real-time CUDA renderer (Windows): multiscale sphere tracing, GEMM normals, normal/texture mapping, FPS benchmark |
| `attributes/` | Neural texture training on SDF level-set neighborhoods |
| `tools/` | Data acquisition/preparation, archive builder |
| `REPRODUCING.md` | Paper table/figure → command map, with hardware notes |

## Setup

```bash
conda env create -f environment.yml     # env: new_i3d
conda activate new_i3d
pip install -e .
```

The evaluation protocol, the renderer, and the texture code have their own
environment notes (`metrics/`, `renderer/README.md`,
`attributes/README.md`).

## Data

Download `m-plicits-data.zip` (link: **TODO on release**) and unzip it at
the repository root — it provides the training/evaluation point clouds,
noisy and outlier-contaminated variants, released model checkpoints, and
the renderer/texture assets, in place. Details and checksums:
`data/README.md`.

## Quickstart

```bash
# train the three stages on a shape
python train_sdf.py data/normalized_sphere/input/normalized_armadillo_gt.ply results/normalized_armadillo_gt/coarse experiments/armadillo_coarse.yaml
python experiment_scripts/train_sdf_on_neighborhood.py data/normalized_sphere/input/normalized_armadillo_gt.ply results/normalized_armadillo_gt/medium experiments/arm_res_on_neigh_medium.yaml results/normalized_armadillo_gt/coarse/best.pth
python experiment_scripts/train_sdf_on_neighborhood_fine.py data/normalized_sphere/input/normalized_armadillo_gt.ply results/normalized_armadillo_gt/fine experiments/arm_res_on_neigh_fine.yaml results/normalized_armadillo_gt/coarse/best.pth results/normalized_armadillo_gt/medium/best.pth

# multiscale band-culled reconstruction
python reconstruct.py results/normalized_armadillo_gt/fine/best.pth out.ply -r 512 --multistage \
    --coarse_path results/normalized_armadillo_gt/coarse/best.pth --medium_path results/normalized_armadillo_gt/medium/best.pth

# real-time rendering (after building renderer/ — see renderer/README.md)
renderer/cuda/build/Release/MIP-plicitsRenderer.exe -experiment=armadillo
```

Every paper and appendix result maps to a command in
[REPRODUCING.md](REPRODUCING.md).

## Citation

```bibtex
% TODO on acceptance
```

## License

MIT (see `LICENSE`); vendored third-party components under their own terms
(`THIRD_PARTY_NOTICES.md`).
