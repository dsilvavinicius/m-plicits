# Real-time renderer

CUDA implementation of multiscale sphere tracing with GEMM-based analytical
normals and neural normal/texture mapping — the source of the paper's
real-time results (Tab. 4 renderer ablation, the FPS columns of Tab. 2, and
the rendering figures). Windows-only (freeglut/GLEW binaries and OpenGL
interop; tested with Visual Studio 2022, CUDA ≥ 12.1, CMake ≥ 3.23).

## Build

Checkpoints come from the data archive (see `../data/README.md`), which
unpacks into `cuda/data/`. Then:

```bash
cd renderer/cuda
cmake -S . -B build              # options below
cmake --build build --config Release
```

Options:
- `-DMIP_CUDA_ARCH=86` (default) embeds SASS+PTX for Ampere; newer GPUs run
  it via JIT. Set your native arch (e.g. `120` for an RTX 5090) for the
  fastest first launch.
- `-DMIP_RESOLUTION=512|1024` — compiled in, because the CUTLASS GEMM tile
  shapes are resolution-specific (`layer_gemms.cu`).
- `-DMIP_BUILD_TESTS=ON` for the GoogleTest target.

## Run

```bash
cd build/Release        # data/ and the DLLs are copied here post-build
MIP-plicitsRenderer.exe -list                      # available experiments
MIP-plicitsRenderer.exe -experiment=armadillo      # interactive
MIP-plicitsRenderer.exe -experiment=lucy -benchmark=500
```

- `-experiment=<name>` selects the model set (default `armadillo`); the
  ImGui panel can also switch between the main shapes at runtime.
- `-data_root=<dir>` points at a checkpoint folder other than `./data`.
- `-benchmark=<N>` renders N frames after a 30-frame warmup, prints
  `BENCHMARK resolution=... iters=... avg_fps=...`, appends the
  configuration and FPS to `benchmark.csv`, dumps the last frame as
  `img<N>.ppm`, and exits — this reproduces the Tab. 4 protocol (previously
  the FPS was read manually from the window title). Benchmark mode redraws
  continuously; the interactive window refreshes on a 10 ms timer and is
  therefore capped near 100 FPS whatever the GPU cost.

Sphere-tracing configuration (command line; the same values are ImGui
sliders in the interactive window). **The defaults trace the coarse level
only** (20 iterations), which is the LoD-0 row of Tab. 4, not the full
model:

| Flag | Meaning | Default |
|---|---|---|
| `-iters=<c>,<m>,<f>` | sphere-tracing iterations on the coarse SDF, then with the medium and fine residuals added | `20,0,0` |
| `-delta=<d0>[,<d1>]` | coarse band half-width: rays trace the offset surface `f1 = d0` and switch to the finer levels within `1.4·d0` of it (the paper's δ₁; `0` never hands over to the finer levels) | `0` |
| `-normal_lod=<0|1|2>` | which level's analytic normals shade the traced point — neural normal mapping when it exceeds the last traced level | `0` |
| `-threshold=<t>` | hit threshold on the summed distance | `0.05` |
| `-shading=phong|normals` | Phong shading or normals as colour | `phong` |
| `-skip_lod0`, `-no_residual` | skip the coarse pass / treat the levels as independent SDFs instead of residuals | off |

The paper's rows, for `armadillo`, `lucy`, `buddha` and `thai_statue`:

```bash
MIP-plicitsRenderer.exe -experiment=armadillo -benchmark=500                                                 # coarse only
MIP-plicitsRenderer.exe -experiment=armadillo -benchmark=500 -iters=20,5,5 -delta=0.02 -normal_lod=2         # full detail
MIP-plicitsRenderer.exe -experiment=armadillo -benchmark=500 -iters=20,5,0 -delta=0.02 -normal_lod=2         # medium surface, fine normals (NM)
```

Note that `-normal_lod` matters for the full-detail row too: tracing the
finer levels refines the silhouette, but most of the visible detail comes
from shading with the finer level's normals, so `-iters=20,5,5` with the
default `-normal_lod=0` looks coarse.

`-delta` should be the coarse band width the model was trained with
(Eq. 5 of the paper; `appendix_experiments/extract_deltas.py` prints it for
a checkpoint). Too small and the finer levels never engage; too large and
rays stop short of the surface.

Each experiment (architecture per LoD + checkpoint paths + orientation) is
declared in `state.cu`'s registry; add new shapes there. The camera starts
3.5 units back so the whole normalized object is visible; rays are clipped
to the scene's bounding sphere before tracing (SIREN SDFs are only reliable
near the [-1,1]³ training domain). If a checkpoint's native axes render the
model upside-down, set `flip_y = true` in its experiment declaration.

## Checkpoint format

The renderer reads flat little-endian float32 `.bin` pairs
(`<stem>_weights.bin`, `<stem>_biases.bin`) with **no header** — layer
sizes come from the experiment declaration. Convert a training `.pth`:

```bash
python ../scripts/weights_biases_from_pth.py -f data/<model>.pth
```

The archive ships both the `.pth` and the derived `.bin` files. Inference
runs in fp16 (`cutlass::half_t`).

## GEMM tuning (optional)

`../scripts/{setup,download,build,profile}_gemms.sh` drive the CUTLASS 2.8
profiler over the exact GEMM shapes used per resolution; the winning
template parameters are transcribed into `layer_gemms.cu`. The shipped
parameters were tuned on RTX 2070/3090-class GPUs and run well on newer
hardware; re-tune for maximum FPS on other architectures.

## Python reference renderer

`../python/` contains the (orders-of-magnitude slower) PyTorch reference
implementation used for the comparison columns: `mip-plicits.py`,
`siren-mips.py`, `siren-mips-times.py`, animation variants, and
`bacon-mips.py` (requires BACON's `modules.py` from its official
repository, placed on `PYTHONPATH`). They use the `i3d` package from the
repository root (`pip install -e ..`).
