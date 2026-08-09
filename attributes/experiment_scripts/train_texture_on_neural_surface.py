#!/usr/bin/env python
# coding: utf-8

import sys
import os
import json
import torch
from torch.utils.data import DataLoader
import configargparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio
import utils
import training
import loss_functions
from model import SIREN


def parse_list_cmd_arg(cmdarg, sep=",", typ=int):
    parsed_arg = cmdarg
    if not isinstance(parsed_arg, list):
        parsed_arg = parsed_arg.split(sep)
    for i in range(len(parsed_arg)):
        parsed_arg[i] = typ(parsed_arg[i])
    return parsed_arg


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1400)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--net_config', nargs="+", type=int, required=True, help='Network configuration')

p.add_argument('--w0', type=int, default=30,
               help='Multiplicative factor for the frequencies')
p.add_argument('--mesh_path', type=str, default='./data/armadillo.ply',
               help='Mesh input path')
p.add_argument('--shapeNet_path', type=str, default='./shapeNets/bob_1x64_w0-16.pth',
               help='shapeNet input path')
p.add_argument('--shapeNet_w0', type=int, default=30,
               help='omega_0 of the pretrained SDF network (matches the checkpoint, e.g. 16 for *_w0-16.pth).')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

available = torch.cuda.is_available()
print(f"CUDA available? {available}", flush=True)

device = torch.device("cuda:0" if available else "cpu")
print(f"Device: {device}")

# launch the trained model
trained_model = utils.from_pth(opt.shapeNet_path, w0=opt.shapeNet_w0, device=device)

sdf_dataset = dataio.PointCloudMeshTexture(
    trained_model,
    opt.mesh_path,
    batch_size=opt.batch_size
)

dataloader = DataLoader(
    sdf_dataset,
    shuffle=True,
    batch_size=1,
    pin_memory=False,
    num_workers=0,
)

# Define the local model.
texNet = SIREN(3, 3, opt.net_config, w0 = opt.w0)
texNet.to(device)

print(f"Is model on GPU? {next(texNet.parameters()).is_cuda}", flush=True)

# Define the loss
loss_fn = loss_functions.loss_texture(trained_model)
summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(
    model=texNet,
    train_dataloader=dataloader,
    epochs=opt.num_epochs,
    lr=opt.lr,
    steps_til_summary=opt.steps_til_summary,
    epochs_til_checkpoint=opt.epochs_til_ckpt,
    model_dir=root_path,
    loss_fn=loss_fn,
    summary_fn=summary_fn,
    clip_grad=True,
    device=device
)
