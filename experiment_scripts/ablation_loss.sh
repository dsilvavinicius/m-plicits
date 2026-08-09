#!/bin/bash

EXPERIMENTS=$(ls $1 | grep medium | sort)
for exp in ${EXPERIMENTS[@]}; do
    echo "========== Running for ${exp} ==========="
    if [ -f results/loss_ablation/${exp%%.*}/best.pth ]; then
        continue
    fi
    python experiment_scripts/train_sdf_on_neighborhood.py data/lucy_simple.ply results/loss_ablation/${exp%%.*} $1/${exp} --device cuda:1
    # python reconstruct.py results/loss_ablation/${exp%%.*}/best.pth --coarse_path data/lucy_64x1.pth results/loss_ablation/${exp%%.*}/best.ply --resolution 400
done

EXPERIMENTS_FINE=$(ls $1 | grep fine | sort)
for exp in ${EXPERIMENTS_FINE[@]}; do
    echo "========== Running for ${exp} ==========="
    if [ -f results/loss_ablation/${exp%%.*}/best.pth ]; then
        continue
    fi
    python experiment_scripts/train_sdf_on_neighborhood_fine.py data/lucy_simple.ply results/loss_ablation/${exp%%.*} $1/${exp} --device cuda:1
    # python reconstruct.py results/loss_ablation/${exp%%.*}/best.pth --coarse_path data/lucy_mipplicits/64x1.pth --medium_path data/lucy_mipplicits/256x1.pth results/loss_ablation/${exp%%.*}/best.ply --resolution 400
done
