# M-plicits: Neural Implicit Surfaces via Nested Multiscale Residuals

[Project page](https://dsilvavinicius.github.io/m-plicits) ·
Paper (NeurIPS 2026; link added on publication) ·
[Data & models](https://github.com/dsilvavinicius/m-plicits/releases)

Official code release. M-plicits models a signed distance function as a
base SIREN plus a sequence of residual SIRENs, each supervised only inside
the adaptive δ-band of the previous level's zero-level set. The nesting is
a training-time invariant that the inference algorithms exploit: multiscale
sphere tracing, band-culled marching cubes, GEMM-based analytical normals,
and neural normal/texture mapping — real-time rendering from a compact
model, with strong robustness to input noise.

## Repository map

| Folder / file | Contents |
|---|---|
| `reproduce.py` | One driver for the paper's tables from the released checkpoints (`python reproduce.py all`) |
| `i3d/`, `train_sdf.py`, `experiment_scripts/`, `reconstruct.py` | Training: coarse stage + banded residual stages; multiscale, band-culled reconstruction |
| `results/<shape>/<stage>/` (from the data archive) | Released checkpoints **and the exact config that trained them** |
| `appendix_experiments/` | Drivers for every appendix experiment (isolation ablation, SPSR, BANF re-implementation, robustness studies) |
| `metrics/` | The paper's evaluation protocol (CD / Hausdorff / voxel IoU) + baseline harnesses |
| `renderer/` | Real-time CUDA renderer (Windows): multiscale sphere tracing, GEMM normals, normal/texture mapping, FPS benchmark, runtime loading of any released model |
| `attributes/` | Neural texture training on SDF level-set neighborhoods |
| `tools/` | `download_data.py`, data preparation, archive builder |
| `docs/` | The project page (GitHub Pages) |
| `REPRODUCING.md` | Table/figure → command map, with the verification log |

## Setup

```bash
conda env create -f environment.yml     # env: new_i3d (Python 3.10, PyTorch 2.7.1 + CUDA 12.8 wheels)
conda activate new_i3d
pip install -e .
```

On Linux a plain virtualenv with `pip install -r requirements.txt` is
equivalent. The evaluation metrics need PyTorch3D and live in their own
environment (`metrics/environment.yml`); the renderer (`renderer/README.md`)
and the texture code (`attributes/README.md`) have their own notes.

## Data

```bash
python tools/download_data.py                 # inputs + released models (1.2 GB), unpacked in place
python tools/download_data.py --paper-meshes  # optional: the paper's noise reconstructions (0.5 GB)
```

The archive recreates `data/`, `results/`, `renderer/cuda/data/` and
`attributes/{data,shapeNets}` at the repository root; `data/README.md`
describes the layout, and every file is listed with a checksum inside the
archive. `--gdrive <id>` switches to a Google Drive mirror.

## Reproduce the paper

```bash
python reproduce.py all          # Tab. 2 (ours), extraction times, Tab. 4 (if the renderer is built) -> reproduction/REPORT.md
python reproduce.py paper-noise  # Tab. 3 (ours) from the paper's reconstructions, no retraining
python reproduce.py train --shapes normalized_armadillo_gt   # retrain a shape from scratch with the shipped config
python reproduce.py noise --shapes normalized_armadillo_gt   # retrain on its 1% noisy input (Tab. 3 protocol)
```

Every stage prints the reproduced number next to the published one. What
was verified, on which hardware, and how far each number moved is recorded
in [REPRODUCING.md](REPRODUCING.md).

## Quickstart on one shape

The configuration that produced each released checkpoint ships next to it,
and the trainers read it directly:

```bash
S=normalized_armadillo_gt
python train_sdf.py data/normalized_sphere/input/$S.ply out/$S/coarse results/$S/coarse/config.yaml
python experiment_scripts/train_sdf_on_neighborhood.py data/normalized_sphere/input/$S.ply out/$S/medium results/$S/medium/config.yaml out/$S/coarse/best.pth
python experiment_scripts/train_sdf_on_neighborhood_fine.py data/normalized_sphere/input/$S.ply out/$S/fine results/$S/fine/config.yaml out/$S/coarse/best.pth out/$S/medium/best.pth

# multiscale, band-culled reconstruction at 512^3
python reconstruct.py out/$S/fine/best.pth out/$S/mesh.ply -r 512 --device cuda --multistage --coarse_path out/$S/coarse/best.pth --medium_path out/$S/medium/best.pth
```

(`experiments/*.yaml` are older exploratory configs, kept for the appendix
ablations; they are not the released models.)

## Real-time renderer

Build per `renderer/README.md`, then from `renderer/cuda/build/Release`:

```bash
MIP-plicitsRenderer.exe -experiment=armadillo -iters=20,5,5 -delta=0.02 -normal_lod=2    # full detail, interactive
MIP-plicitsRenderer.exe -experiment=armadillo -benchmark=500                              # coarse-only FPS row of Tab. 4

# any released model, without touching the registry:
python ../../scripts/export_experiment.py ../../../../results/normalized_lucy_gt --out data/released --flip-y
MIP-plicitsRenderer.exe -experiment_file=data/released/normalized_lucy_gt.exp -iters=20,5,5 -delta=0.02 -normal_lod=2
```

## Citation

```bibtex
@inproceedings{mplicits2026,
  title     = {M-plicits: Neural Implicit Surfaces via Nested Multiscale Residuals},
  author    = {da Silva, Vin{\'\i}cius and Melo, Isabelle and Bessa, Matheus and
               Schardong, Guilherme and Schirmer, Luiz and Ara{\'u}jo, Andr{\'e} and
               Gon{\c{c}}alves, Nuno and Lopes, H{\'e}lio and Raposo, Alberto and
               Velho, Luiz and Novello, Tiago},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`); vendored third-party components under their own terms
(`THIRD_PARTY_NOTICES.md`).
