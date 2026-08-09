#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
import os
import os.path as osp
import yaml
from i3d.loss_functions import loss_add_detail

BASE_CFG_PATH_FMT = osp.join("experiments", "{}_on_neigh_{}.yaml")
MESHES = ["lucy"]
LEVELS = ["medium", "fine"]
OUTPUT_CFG_DIR = osp.join("experiments", "loss_ablation")

if not osp.exists(OUTPUT_CFG_DIR):
    os.makedirs(OUTPUT_CFG_DIR)

for mesh in MESHES:
    for level in LEVELS:
        cfgpath = BASE_CFG_PATH_FMT.format(mesh, level)
        with open(cfgpath, 'r') as f:
            base_cfg = yaml.safe_load(f)

        if "constraint_weights" not in base_cfg["training"]:
            base_cfg["training"]["constraint_weights"] =\
                deepcopy(loss_add_detail.DEFAULT_WEIGHTS)

        sdf_constraint_vals = [0, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4]
        for v in sdf_constraint_vals:
            cfg = deepcopy(base_cfg)
            cfg["training"]["constraint_weights"]["sdf_constraint"] = v

            with open(osp.join(OUTPUT_CFG_DIR, f"{mesh}_{level}_sdf_{v}.yaml"), 'w') as f:
                yaml.dump(cfg, f)

        normal_constraint_vals = [0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3]
        for v in sdf_constraint_vals:
            cfg = deepcopy(base_cfg)
            cfg["training"]["constraint_weights"]["normal_constraint"] = v

            with open(osp.join(OUTPUT_CFG_DIR, f"{mesh}_{level}_normal_{v}.yaml"), 'w') as f:
                yaml.dump(cfg, f)

        grad_constraint_vals = [0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3]
        for v in sdf_constraint_vals:
            cfg = deepcopy(base_cfg)
            cfg["training"]["constraint_weights"]["grad_constraint"] = v

            with open(osp.join(OUTPUT_CFG_DIR, f"{mesh}_{level}_grad_{v}.yaml"), 'w') as f:
                yaml.dump(cfg, f)
