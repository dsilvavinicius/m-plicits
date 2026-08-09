'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import utils
import meshing
import configargparse
import diff_operators

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--resolution', type=int, default=256)

p.add_argument('--shape_net_path', type=str, required=True)

p.add_argument('--shape_net_w0', type=int, required=True)

p.add_argument('--w0', type=int, required=True, help='Multiplicative factor for the frequencies')

opt = p.parse_args()

in_features = 3

sdf_decoder1 = utils.from_pth(opt.shape_net_path, w0=opt.shape_net_w0)
sdf_decoder1.cuda()

sdf_decoder2 = utils.from_pth(opt.checkpoint_path, w0=opt.w0)
sdf_decoder2.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

meshing.create_mesh(sdf_decoder1, sdf_decoder2,  os.path.join(root_path, 'test'), N=opt.resolution)
