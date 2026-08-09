#!/usr/bin/env python
# coding: utf-8

import sys
import os
import json
import torch
from torch.utils.data import DataLoader
import configargparse
import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import Dataset
from utils import from_pth, write_sdf_summary
import training
import loss_functions
from model import SIREN
import diff_operators


def parse_list_cmd_arg(cmdarg, sep=",", typ=int):
    parsed_arg = cmdarg
    if not isinstance(parsed_arg, list):
        parsed_arg = parsed_arg.split(sep)
    for i in range(len(parsed_arg)):
        parsed_arg[i] = typ(parsed_arg[i])
    return parsed_arg


class PointCloudMeshTexture(Dataset):
    def __init__(self, trained_model, mesh_path, batch_size=0, silent=False):
        super().__init__()

        self.shapeNet = trained_model
        self.shapeNet.cuda()

        # coding: utf-8

        # Load the point cloud
        point_cloud = o3d.io.read_point_cloud(mesh_path)

        # Convert the point cloud to a numpy array for easier handling
        self.points  = torch.Tensor(np.asarray(point_cloud.points )).to(device)
        self.normals = torch.Tensor(np.asarray(point_cloud.normals)).to(device)
        self.colors  = torch.Tensor(np.asarray(point_cloud.colors )).to(device)

        self.batch_size = batch_size

        print("Done preparing the dataset.")

    def __len__(self):
        #lenght = self.surface_samples.size(0) // (self.batch_size) + 1
        return 1#lenght

    def __getitem__(self, idx):
        
        idx = np.random.choice(self.points.shape[0], self.batch_size)
        coords = self.points[idx, ...]
        normals = self.normals[idx, ...]
        textures = self.colors[idx, ...]

        #projecting the coords to the neural surface
        trained_model = self.shapeNet(coords.cuda())
        coords = trained_model['model_in']
        sdf    = trained_model['model_out']

        grad = diff_operators.gradient(sdf, coords)
        grad_norm = torch.norm(grad, dim=-1)
        unit_grad = grad/grad_norm.unsqueeze(-1)
        coords = coords - sdf*unit_grad
        coords = coords.detach()
        
        # # Create a pointclouds object with positions and colors.
        # pointclouds = pytorch3d.structures.Pointclouds(points=[coords.squeeze(0)], features=[textures.squeeze(0)])

        # # Plot the point cloud using the default AxisArgs and PointcloudsVisualizer.
        # fig = plot_batch_individually([pointclouds], subplot_titles=["plot1"])
        # fig.show()

        return  {
                     "coords" : coords.float()
                }, {
                    "normals" : normals.float(),
                    "textures": textures.float()
                }
        #return self._random_sampling(self.batch_size)

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.surface_samples.size(0)

        on_surface_count = n_points
        
        on_surface_samples = []

        idx = np.random.choice(self.surface_samples.size(0), on_surface_count)
        on_surface_samples = self.surface_samples[idx, ...]
        
        samples = on_surface_samples

        #projecting the coords to the neural surface
        coords = samples[:, :3].float().cuda()
        trained_model = self.shapeNet(coords)
        coords = trained_model['model_in']
        sdf    = trained_model['model_out']

        grad = diff_operators.gradient(sdf, coords)
        # grad_norm = torch.norm(grad, dim=-1)
        # unit_grad = grad/grad_norm.unsqueeze(-1)
        coords = coords - sdf*grad
        coords = coords.cpu().detach()
        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": coords.float()
        }, {
            "normals": samples[:, 3:6].float(),
            "textures": samples[:, 6:9].float()
        }


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
               help='omega_0 of the pretrained SDF network (matches the checkpoint).')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

available = torch.cuda.is_available()
print(f"CUDA available? {available}", flush=True)

device = torch.device("cuda:0" if available else "cpu")
print(f"Device: {device}")

# launch the trained model
trained_model = from_pth(opt.shapeNet_path, w0=opt.shapeNet_w0, device=device)

sdf_dataset = PointCloudMeshTexture(
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
summary_fn = write_sdf_summary

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
