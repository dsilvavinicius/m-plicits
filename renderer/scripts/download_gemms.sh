#!/bin/bash

# Downloads CUTLASS 2.7.0, creating one copy for each GEMM variant (nn, nt, tn, tt)
# params: $1 is the path to download the GEMM variants.

echo "================================================================="
echo "Starting to download CUTLASS 2.8.0 to $1."
echo -e "=================================================================\n"

mkdir -p "$1"
cd "$1"

curl -OL https://github.com/NVIDIA/cutlass/archive/refs/tags/v2.8.0.tar.gz

create_variant() {
    echo "Extracting CUTLASS to create variant $1."
    mkdir ./cutlass_"$1"
    tar -xf v2.8.0.tar.gz -C cutlass_"$1"
    echo ""
}

create_variant nn
create_variant nt
create_variant tn
create_variant tt

echo "================================================================="
echo "Download finished."
echo -e "=================================================================\n"