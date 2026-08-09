#!/bin/bash

# Profile all GEMM variants (nn, nt, tn, tt)
# Params: $1 is the path to find the GEMM variants.
# $2 is the path to put profiling results.

profile() {
    #params: $1 is the variant sufix (nn, nt, tn or tt).
    # $2 is the path to put profiling results.

    echo "Profiling variant $1."
    cd cutlass_"$1"/build/tools/profiler/Release/
    ./cutlass_profiler --m=8,64,128,256 --n=262144,589824,1048576 --k=3,4,64,128,256 --output="$2"/report_"$1".csv
    cd ../../../../../
    echo ""
}

echo "================================================================="
echo "Starting to profile all GEMM variants (nn, nt, tn and tt) in $1."
echo -e "=================================================================\n"

cd "$1"
mkdir -p "$2"

profile nn "$2"
profile nt "$2"
profile tn "$2"
profile tt "$2"

echo "================================================================="
echo "Profiling finished."
echo -e "=================================================================\n"