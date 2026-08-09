#!/bin/bash

# Build all GEMM variants (nn, nt, tn, tt)
# params: $1 is the path to find the GEMM variants.
# $2 is the CUDA capability for GEMMs (75 for Turing and 80 for Ampere, for example)

build() {
    #params: $1 is the variant sufix (nn, nt, tn or tt).
    echo "Building variant $1."
    cd cutlass_"$1"
    mkdir build
    cd build
    cmake ../cutlass-2.8.0 -DCUTLASS_NVCC_ARCHS="$2" -DCUTLASS_LIBRARY_KERNELS=gemm*"$1" -DCUTLASS_UNITY_BUILD_ENABLED=ON -DCUTLASS_ENABLE_TESTS=OFF
    msbuild.exe CUTLASS.sln //t:Clean,Build //p:Configuration=Release //p:Platform=x64
    cp ./tools/library/Release/cutlass.dll ./tools/profiler/Release/cutlass.dll
    cd ../..
    echo ""
}

echo "================================================================="
echo "Starting to build all GEMM variants (nn, nt, tn and tt) in $1."
echo -e "=================================================================\n"

export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
cd "$1"

build nn "$2"
build nt "$2"
build tn "$2"
build tt "$2"

echo "================================================================="
echo "Building finished."
echo -e "=================================================================\n"