# Third-party notices

This repository vendors or derives from the following projects; their
license terms apply to the corresponding subtrees.

| Component | Location | License |
|---|---|---|
| NVIDIA CUTLASS 2.8 | `renderer/cuda/inc/cutlass`, `renderer/cuda/tools/util` | BSD-3-Clause (NVIDIA) |
| Dear ImGui | `renderer/cuda/imgui` | MIT |
| freeglut | `renderer/cuda/lib`, `renderer/cuda/inc/GL` | MIT/X-Consortium |
| GLEW | `renderer/cuda/lib`, `renderer/cuda/inc/GL` | Modified BSD / MIT |
| CUDA Samples helpers | `renderer/cuda/inc/helper_*.h` | NVIDIA CUDA Samples license |
| glm | fetched at build time (pinned tag 1.0.1) | The Happy Bunny License / MIT |
| GoogleTest | fetched at build time when `MIP_BUILD_TESTS=ON` (v1.14.0) | BSD-3-Clause |
| SIREN (V. Sitzmann) | `attributes/` derives from the SIREN codebase | MIT (see `attributes/LICENSE`) |
| DeepSDF (Meta) | `attributes/meshing.py` mesh export derives from DeepSDF | MIT |
| Keenan Crane's model repository | spot/bob/blub meshes in the data archive | CC0 |
| Stanford 3D Scanning Repository / Thingi10K | scan-derived point clouds in the data archive | respective dataset terms |
