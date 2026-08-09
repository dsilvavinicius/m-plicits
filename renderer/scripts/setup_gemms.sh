#!/bin/bash

# Download, build and profile all GEMMs. Usefull for finding the correct GEMM parameters for a machine.
# Params: $1 is the path to install the GEMM variants.
# $2 is the path to put profiling results.
# $3 is the CUDA capability for GEMMs (75 for Turing and 80 for Ampere, for example)

echo "================================================================="
echo "Starting to setup GEMMs at $1"
echo "Profiling results will be at $2"
echo "CUDA capability will be $3"
echo -e "=================================================================\n"

./download_gemms.sh "$1"
./build_gemms.sh "$1" "$3"
./profile_gemms.sh "$1" "$2"