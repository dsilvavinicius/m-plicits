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
  `BENCHMARK resolution=... avg_fps=...`, appends to `benchmark.csv`, and
  exits — this reproduces the Tab. 4 protocol (previously the FPS was read
  manually from the window title). Sphere-tracing iteration counts and the
  LoD shown are controlled in the ImGui panel; Tab. 4 uses 20 iterations on
  the first SDF and 5 per subsequent level, at 512².

Each experiment (architecture per LoD + checkpoint paths + orientation) is
declared in `state.cu`'s registry; add new shapes there.

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
