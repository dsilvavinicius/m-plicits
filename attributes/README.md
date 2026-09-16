# Neural texture (attribute) mapping

Trains an RGB SIREN on the neighborhood of a pretrained neural SDF's zero
level set, reproducing the **texture half** of the paper's attribute-mapping
supplementary section (the texture figures and the texture MSE table).
Neural **normal** mapping is not trained — it is the analytic gradient
`g(p) = ∇f(p)` of a finer SDF, evaluated by the real-time renderer (see
`../renderer`).

## Environment

```bash
conda env create -f environment.yml   # env: neural_textures
conda activate neural_textures
pip install --no-build-isolation -r ../metrics/requirements-pytorch3d.txt   # PyTorch3D from source (Windows; Linux can use the pytorch3d conda channel)
```

Verified end to end on 2026-09-16 (train 100 epochs on `spot`, extract the
coloured mesh) on Windows 11 / RTX 5090. If `import mesh_to_sdf` dies
silently on Windows, reinstall numpy (`pip install --force-reinstall --no-deps numpy==2.2.6`):
a damaged BLAS DLL from a mixed conda/pip install crashes at the first
matrix product.

## Data

Shipped in the data archive (see `../data/README.md`): textured meshes under
`attributes/data/` (spot, bob, blub, bunny, earth) and pretrained SDF
checkpoints under `attributes/shapeNets/`. The ω₀ of each SDF checkpoint is
encoded in its filename (e.g. `spot_1x64_w0-16.pth` → `--shapeNet_w0 16`)
and **must** be passed explicitly — the loader cannot recover it from the
file.

## Usage

Train a texture on a UV-textured mesh:

```bash
python experiment_scripts/train_texture_on_neural_surface.py \
    --experiment_name spot --net_config 256 256 256 --w0 120 \
    --shapeNet_path ./shapeNets/spot_1x64_w0-16.pth --shapeNet_w0 16 \
    --mesh_path ./data/spot/spot/spot_tex.obj \
    --batch_size 1400 --lr 1e-4 --num_epochs 10000 --logging_root ./logs
```

(`train_texture_on_neural_surface_pc.py` is the colored-point-cloud
variant.) Extract a colored mesh from the trained pair:

```bash
python tools/run_test.py --base_dir ./logs/spot \
    --shape_net_path ./shapeNets/spot_1x64_w0-16.pth --shape_net_w0 16 \
    -w0 120 -c final -r 256
```

Reproduce a row of the texture MSE table:

```bash
python experiment_scripts/mse_comparison.py \
    --obj ./data/earth_hallison/Terra.obj \
    --recon ./results/terra/reconstructions/current.ply \
    --sdf-pth ./shapeNets/terra_w0_16.pth --sdf-w0 16 \
    --tex-pth ./results/terra/checkpoints/model_final.pth --tex-w0 120
```

Note: the "Egg" row of the MSE table cannot be reproduced from released
assets — the egg mesh/texture is not part of the archive.

## Provenance

Derived from Vincent Sitzmann's SIREN codebase (see LICENSE); the mesh IO
and marching-cubes export derive from DeepSDF. This release ships only the
working texture pipeline; the fork's unrelated experimental scripts were
removed.
