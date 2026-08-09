#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import os.path as osp
import pandas as pd
import torch
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import time
from kornia.metrics import psnr
from i3d.util import from_pth
import numpy as np
import cv2

def init_ray_positions(n_pixels, experiment_name, cam_time, device="cuda:0"):
    """Initializes the ray positions according to the experiment.

    Parameters
    ----------
    n_pixels: int
        Viewport width in pixels. We assume a square viewport, thus, the
        resulting tensor will be shaped (n_pixels^2, 3).

    experiment_name: str
        The experiment name. May be any string. For "lucy*", "buddha*" and,
        "bunny*", we employ slightly different initialization mechanisms.

    device: str, optional
        The device to store our tensor. By default its the first CUDA
        capable GPU.

    Returns
    -------
    pos: torch.Tensor
        The initial ray position tensor with shape (n_pixels^2, 3).
    """
    pos = torch.zeros((n_pixels * n_pixels, 3), requires_grad=True,
                      device=torch.device(device))
    
    

    with torch.no_grad():
        x = torch.linspace(-1, 1, steps=n_pixels)
        xs, ys = torch.meshgrid(x, x, indexing='ij')

        if "lucy" in experiment_name:
            pos[..., 0] = xs.reshape(-1, 1).squeeze()
            pos[..., 1] = 0.6 * torch.ones_like(pos[..., 0])
            pos[..., 2] = ys.reshape(-1, 1).squeeze()

            x = pos[..., 0].clone()
            y = pos[..., 1].clone()
            pos[..., 0] =  np.cos(cam_time) * x + np.sin(cam_time) * y
            pos[..., 1] = -np.sin(cam_time) * x + np.cos(cam_time) * y

        elif "buddha" in experiment_name or "bunny" in experiment_name or "dragon" in experiment_name:
            pos[..., 0] = xs.reshape(-1, 1).squeeze()
            pos[..., 1] = ys.reshape(-1, 1).squeeze()
            pos[..., 2] =  torch.ones_like(pos[..., 1])
    
            x = pos[..., 0].clone()
            y = pos[..., 2].clone()
            pos[..., 0] =  np.cos(cam_time) * x + np.sin(cam_time) * y
            pos[..., 2] = -np.sin(cam_time) * x + np.cos(cam_time) * y
        else:
            pos[..., 0] = xs.reshape(-1, 1).squeeze()
            pos[..., 1] = ys.reshape(-1, 1).squeeze()
            pos[..., 2] = -torch.ones_like(pos[..., 1])*1.05

            x = pos[..., 0].clone()
            y = pos[..., 2].clone()
            pos[..., 0] =  np.cos(cam_time) * x + np.sin(cam_time) * y
            pos[..., 2] = -np.sin(cam_time) * x + np.cos(cam_time) * y


    return pos


def init_ray_dir(experiment_name, cam_time, device):
    """Initializes the ray directions according to the experiment.

    Parameters
    ----------
    experiment_name: str
        The experiment name. May be any string. For "lucy*", "buddha*" and,
        "bunny*", we employ slightly different initialization mechanisms.

    device: str, optional
        The device to store our tensor. By default its the first CUDA
        capable GPU.

    Returns
    -------
    dir: torch.Tensor
        The ray direction tensor with shape (1, 3).
    """
    if "lucy" in experiment_name:
        dir = torch.tensor([0.0, -1.0, 0.0], device=device)

        dir[...,0]=-np.sin(cam_time)
        dir[...,2]=0	
        dir[...,1]=-np.cos(cam_time)
    elif "bunny" in experiment_name or "buddha" in experiment_name or "dragon" in experiment_name:
        dir = torch.tensor([0.0, 0.0, -1.0], device=device)
        
        dir[...,0]=-np.sin(cam_time)
        dir[...,1]=0	
        dir[...,2]=-np.cos(cam_time)
    else:
        dir = torch.tensor([0.0, 0.0, 1.0], device=device)

        dir[...,0]=np.sin(cam_time)
        dir[...,1]=0	
        dir[...,2]=np.cos(cam_time)

    # directions[idx_directions] = -sin(cam_time);
    # directions[idx_directions + 1] = 0.f;
    # directions[idx_directions + 2] = (invert_z) ? -cos(cam_time) : cos(cam_time);

    return dir

def multiscale_sphere_tracing(
    models, experiment_name, n_pixels, deltas, num_iterations, lod=0,
    multiscale=True, normal_mapping=True,
    dist_threshold=1e-3, n_frames = 100, output_dir="results",
    device="cuda:0"):
    """Performs sphere tracing on a list of Implicit models."""
    time.perf_counter()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_dir + f'/{experiment_name}_LOD_{lod}_multiscale_{multiscale}_mapping_{normal_mapping}.mp4', fourcc, 60.0, (n_pixels,n_pixels))
    for frame in range(n_frames):

        cam_time = -2.*torch.pi*(frame/n_frames)

        n_lods = len(models)
        pos = init_ray_positions(n_pixels, experiment_name, cam_time, device=device)

        with torch.no_grad():
            dir = init_ray_dir(experiment_name, cam_time, device)

            distances = torch.zeros((pos.shape[0], 1), device=device)
            transform = T.ToPILImage()

            start_time = time.perf_counter()

            # multiscale sphere tracing loop
            if multiscale:
                lod_init = 0
            else:
                lod_init = lod

            for i in range(lod_init, lod + 1):
                for j in tqdm(range(0, num_iterations[i]), f"Sphere tracing - LOD {i}"):
                    # infer distance
                    out = models[i]({"coords": pos})['model_out']
                    if i == lod:
                        distances = out
                    else:
                        distances = out - deltas[i]

                    # update pos
                    # pos = torch.clamp(
                    #     pos + torch.mul(distances, dir),
                    #     -1 * torch.ones(1).to(device),
                    #     torch.ones(1).to(device)
                    # )
                    pos = pos + torch.mul(0.6*distances, dir)

            D = torch.cat((distances, distances, distances), dim=-1)
            
            pos_r = torch.ones_like(D)*torch.abs(pos[...,0]).unsqueeze(-1)
            pos_g = torch.ones_like(D)*torch.abs(pos[...,1]).unsqueeze(-1)
            pos_b = torch.ones_like(D)*torch.abs(pos[...,2]).unsqueeze(-1)
            cube_size = 1.1
            mask = D.le(dist_threshold) & torch.le(pos_r, cube_size) & torch.le(pos_g, cube_size) & torch.le(pos_b, cube_size)
        
        result = None
        if normal_mapping:
            result = models[-1]({"coords": pos})
            pbar_str = f"Neural Implicits Normal Mapping - Mapping LOD {n_lods - 1} onto LOD {lod}"
        else:
            result = models[lod]({"coords": pos})
            pbar_str = f"Normals LOD {lod}"

        distances = result["model_out"]
        coords = result["model_in"]

        # Normals.
        with tqdm(total=100, desc=pbar_str) as pbar:
            grad = torch.autograd.grad(
                distances,
                coords,
                grad_outputs=torch.ones_like(distances)
            )[0]

            pbar.update(70)

            with torch.no_grad():
                # Normalization and distance condition
                grad_norm = torch.linalg.norm(grad, dim=-1)

                unit_grad = grad / grad_norm.unsqueeze(-1)

                pbar.update(30)

        # Rendering
        with torch.no_grad():
            with tqdm(total=100, desc="Rendering") as pbar:
                k_s = 0.5
                k_d = 1.0
                k_a = 1.0
                shininess = 35

                ambient = torch.tensor([0.2, 0.2, 0.2]).to(device)
                specular = torch.tensor([1.0, 1.0, 1.0]).to(device)
                diffuse = torch.tensor([0.54, 0.54, 0.54]).to(device)

                light_dir = -dir
                pbar.update(20)

                n = unit_grad
                dot_d = torch.matmul(n, light_dir)
                dot_d = torch.unsqueeze(dot_d, -1)

                reflection = 2.0 * dot_d * n - light_dir

                pbar.update(40)

                dot_d = torch.maximum(dot_d, torch.zeros_like(dot_d))

                dot_s = torch.matmul(reflection, -dir)
                dot_s = torch.unsqueeze(dot_s, -1)
                dot_s = torch.maximum(dot_s, torch.zeros_like(dot_s))**shininess

                color = k_a * ambient + k_d * diffuse * dot_d  + k_s * specular * dot_s
                color = torch.clamp(color, min=0,max=1.0)

                color = torch.where(mask, color, torch.ones_like(color))

                pbar.update(40)

            image_inputs = {"colors" : color}

            for key in tqdm(image_inputs, f"Saving results into {output_dir}"):
                transposed = torch.transpose(image_inputs[key], 0, 1)

                img = transform(torch.reshape(transposed, (3, n_pixels, n_pixels)))
                img = img.rotate(90)

                out_video.write(np.array(img))
    
    out_video.release()
        
    return color


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_folder", default="data", help="Model input folder.",
    )
    parser.add_argument(
        "-o", "--output_folder", default="results/siren",
        help="Results output folder."
    )
    parser.add_argument(
        "-s", "--viewport_size", default=512, type=int,
        help="Width of the output imape in pixels. Note that the image is"
        " always square, so the output image will have SxS pixels."
    )
    parser.add_argument(
        "-d", "--device", default="cuda:0", type=str,
        help="Device to run the tests on. If cuda:0 is selected and no CUDA"
        " capable device is found, we will default to CPU."
    )
    args = parser.parse_args()
    
    if not osp.exists(args.output_folder):
        os.makedirs(args.output_folder)

    devstr = args.device
    if "cuda" in devstr and not torch.cuda.is_available():
        print(f"[WARNING] No CUDA capable device found. Option \"{devstr}\""
              " will be ignored. Running on CPU.")
        devstr = "cpu"

    device = torch.device(devstr)

    experiments = {
        "armadillo": [osp.join(args.input_folder, "armadillo_64x1.pth"),
                                 osp.join(args.input_folder, "armadillo_256x3.pth")],
        #"falcon_witch": [osp.join(args.input_folder, "falcon_witch_2x128_w-20.pth"),
        #                 osp.join(args.input_folder, "falcon_witch_64x1_w0_20_t_-0.2_0.2.pth")],
        # "armadillo_256x3": [osp.join(args.input_folder, "siren_armadillo_256x3.pth")],
        #"buddha": [osp.join(args.input_folder, "siren_buddha_64x1.pth"),
        #                      osp.join(args.input_folder, "siren_buddha_256x3.pth")],
        # "buddha_64x1_256x2_256x3": [osp.join(args.input_folder, "siren_buddha_64x1.pth"),
        #                             osp.join(args.input_folder, "siren_buddha_256x2.pth"),
        #                             osp.join(args.input_folder, "siren_buddha_256x3.pth")],
        # "buddha_256x3": [osp.join(args.input_folder, "siren_buddha_256x3.pth")],
        # "bunny_64x1_256x1_256x3": [osp.join(args.input_folder, "siren_bunny_64x1.pth"),
        #                            osp.join(args.input_folder, "siren_bunny_256x1.pth"),
        #                            osp.join(args.input_folder, "siren_bunny_256x3.pth")],
        #"bunny": [osp.join(args.input_folder, "siren_bunny_64x1.pth"),
        #                    osp.join(args.input_folder, "siren_bunny_256x3.pth")],
        # "bunny_256x3": [osp.join(args.input_folder, "siren_bunny_256x3.pth")],
        # "dragon_64x1_256x1_256x3": [osp.join(args.input_folder, "siren_dragon_64x1.pth"),
        #                             osp.join(args.input_folder, "siren_dragon_256x1.pth"),
        #                             osp.join(args.input_folder, "siren_dragon_256x3.pth")],
        # "dragon_128x1_256x3": [osp.join(args.input_folder, "siren_dragon_128x1.pth"),
        #                        osp.join(args.input_folder, "siren_dragon_256x3.pth")],
        # "dragon_256x3": [osp.join(args.input_folder, "siren_dragon_256x3.pth")],
        #"lucy": [osp.join(args.input_folder, "siren_lucy_64x1.pth"),
        #                    osp.join(args.input_folder, "siren_lucy_256x3.pth")],
        # "lucy_64x1_256x2_256x3": [osp.join(args.input_folder, "siren_lucy_64x1.pth"),
        #                           osp.join(args.input_folder, "siren_lucy_256x2.pth"),
        #                           osp.join(args.input_folder, "siren_lucy_256x3.pth")],
        # "lucy_256x3": [osp.join(args.input_folder, "siren_lucy_256x3.pth")],
    }

    iters = 80

    iterations = {
        "armadillo": [[iters, 40]],
        "bunny": [[iters, 40]],
        "buddha": [[iters, 40]],
        "lucy": [[iters, 40]],
    }

    deltas = {
        "armadillo": [0.008],
        "bunny": [0.008],
        "buddha": [0.008],
        "lucy": [0.025],
    }


    stats_list = []
    for experiment_name in experiments:
        ckpts = experiments[experiment_name]

        print('\n==========\n' + experiment_name.upper() + '\n==========\n')

        models = [None] * len(ckpts)
        for i, c in enumerate(ckpts):
            models[i] = from_pth(c, w0=30, device=device)
            models[i] = models[i].to(device)

        dist_threshold = 5e-2

        print('\n== BASELINE Coarse ==')
        baseline_color = multiscale_sphere_tracing(
            models, experiment_name=experiment_name,
            deltas=deltas[experiment_name], num_iterations=[iters] * len(models),
            n_pixels=args.viewport_size, lod=0,#lod=len(models) - 1,
            multiscale=False, normal_mapping=False,
            dist_threshold=dist_threshold,
            n_frames=300,
            output_dir=args.output_folder,
            device=device
        )

        print('\n== BASELINE Fine ==')
        baseline_color = multiscale_sphere_tracing(
            models, experiment_name=experiment_name,
            deltas=deltas[experiment_name], num_iterations=[iters] * len(models),
            n_pixels=args.viewport_size, lod=1,#lod=len(models) - 1,
            multiscale=False, normal_mapping=False,
            dist_threshold=dist_threshold,
            n_frames=300,
            output_dir=args.output_folder,
            device=device
        )

        N = args.viewport_size
        baseline_color = baseline_color.reshape(3, N, N).unsqueeze(0)

       
        for i in range(len(models)):
            print('===== LODS: ' + str(i + 1)  + ' =====')

            if i == 0:
                print("\n== Normal Mapping ==")
                color_nm = multiscale_sphere_tracing(
                    models, experiment_name=experiment_name,
                    deltas=deltas[experiment_name],
                    num_iterations=[iters] * len(models),
                    n_pixels=args.viewport_size, lod=i,
                    multiscale=False, normal_mapping=True,
                    dist_threshold=dist_threshold,
                    n_frames=300,
                    output_dir=args.output_folder,
                    device=device
                )

            if i > 0:
                for j in range(len(iterations[experiment_name])):
                    print("\n== Multiscale ==")
                    color_ms = multiscale_sphere_tracing(
                        models, experiment_name=experiment_name,
                        deltas=deltas[experiment_name],
                        num_iterations=iterations[experiment_name][j],
                        n_pixels=args.viewport_size, lod=i,
                        multiscale=True, normal_mapping=False,
                        dist_threshold=dist_threshold,
                        n_frames=300,
                        output_dir=args.output_folder,
                        device=device
                    )
