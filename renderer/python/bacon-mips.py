#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import os.path as osp
import sys
import pandas as pd
import torch
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import time
from kornia.metrics import psnr
from i3d.util import from_pth
from modules import MultiscaleBACON
import numpy as np


def calc_deltas(models, n_points: int = 100000, device = torch.device("cuda:0")):
    """Calculates the delta values as per definition 1 in the suplementary material."""

    with torch.no_grad():
        pts = (2 * torch.rand((n_points, 3)) - 1) * 0.95
        pts = pts.to(device)
        eps = [None] * len(models)
        i = 1
        while i < len(models):
            f_i = models[i - 1]({"coords": pts})["model_out"]
            f_ip1 = models[i]({"coords": pts})["model_out"]
            eps[i] = (f_ip1 - f_i).abs().max() # eps_i
            i += 1

        deltas = [None] * len(models)
        deltas[-1] = eps[-1]
        i = len(models) - 2
        while i >= 0:
            deltas[i] = deltas[i + 1] + eps[i + 1]
            i -= 1
    
    return deltas[:-1]


def init_ray_positions(n_pixels, experiment_name, device="cuda:0"):
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
            pos[..., 0] = xs.reshape(-1, 1).squeeze()*0.5
            pos[..., 1] = 0.6 * torch.ones_like(pos[..., 0])
            pos[..., 2] = ys.reshape(-1, 1).squeeze()*0.5
        elif "buddha" in experiment_name or "bunny" in experiment_name or "dragon" in experiment_name:
            pos[..., 0] = xs.reshape(-1, 1).squeeze()*0.5
            pos[..., 1] = ys.reshape(-1, 1).squeeze()*0.5
            pos[..., 2] = 0.6 * torch.ones_like(pos[..., 1])
        else:
            pos[..., 0] = xs.reshape(-1, 1).squeeze()*0.5
            pos[..., 1] = ys.reshape(-1, 1).squeeze()*0.5
            pos[..., 2] = -0.6 * torch.ones_like(pos[..., 1])
    return pos


def init_ray_dir(experiment_name, device):
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
    elif "bunny" in experiment_name or "buddha" in experiment_name or "dragon" in experiment_name:
        dir = torch.tensor([0.0, 0.0, -1.0], device=device)
    else:
        dir = torch.tensor([0.0, 0.0, 1.0], device=device)
    return dir


def multiscale_sphere_tracing(
    model, experiment_name, n_pixels, deltas, num_iterations, lod=0,
    multiscale=True, normal_mapping=True,
    dist_threshold=1e-3, output_dir="results",
    device="cuda:0"):
    """Performs sphere tracing on a list of Implicit models."""
    time.perf_counter()

    n_lods = 2
    pos = init_ray_positions(n_pixels, experiment_name, device=device)

    with torch.no_grad():
        dir = init_ray_dir(experiment_name, device)

        distances = torch.zeros((pos.shape[0], 1), device=device)
        transform = T.ToPILImage()

        start_time = time.perf_counter()

        # multiscale sphere tracing loop
        if multiscale:
            lod_init = 0
        else:
            lod_init = lod

        specified_layers = [2, 6]

        for i in range(lod_init, lod + 1):
            for j in tqdm(range(0, num_iterations[i]), f"Sphere tracing - LOD {i}"):
                # infer distance
                out = model({"coords": pos}, specified_layers[i])['model_out']
                if i == lod:
                    distances = out[0]
                    # print(out)
                else:
                    distances = out[0] - deltas[i]

                # update pos
                pos = torch.clamp(
                    pos + torch.mul(distances, dir),
                    -1 * torch.ones(1).to(device),
                    torch.ones(1).to(device)
                )

            img = transform(distances.reshape(1, n_pixels, n_pixels))
            img = img.rotate(90)
            img.save(osp.join(output_dir, experiment_name + "_lod_" + str(i) + ".png"))

    D = torch.cat((distances, distances, distances), dim=-1)

    result = None
    if normal_mapping:
        result = model({"coords": pos}, specified_layers[-1])
        pbar_str = f"Neural Implicits Normal Mapping - Mapping LOD {n_lods - 1} onto LOD {lod}"
    else:
        result = model({"coords": pos}, specified_layers[lod])
        pbar_str = f"Normals LOD {lod}"


    distances = result["model_out"][0]
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
            unit_grad = torch.abs(unit_grad)

            unit_grad = torch.where(
                D <= dist_threshold,
                unit_grad,
                torch.ones_like(unit_grad)
            )

            pbar.update(30)

    # Rendering
    with torch.no_grad():
        with tqdm(total=100, desc="Rendering") as pbar:
            k_s = 0.5
            k_d = 1.0
            k_a = 1.0
            shininess = 35.0

            ambient = torch.tensor([0.2, 0.2, 0.2]).to(device)
            specular = torch.tensor([1.0, 1.0, 1.0]).to(device)
            diffuse = torch.tensor([0.54, 0.54, 0.54]).to(device)

            if "lucy" in experiment_name:
                light_dir = torch.tensor([0.0, 1.0, 0.0]).to(device)
            else:
                light_dir = torch.tensor([0.0, 0.0, 1.0]).to(device)

            pbar.update(20)

            n = unit_grad
            dot_d = torch.matmul(n, light_dir)
            dot_d = torch.unsqueeze(dot_d, -1)

            reflection = 2.0 * dot_d * n - light_dir

            pbar.update(40)

            dot_d = torch.maximum(dot_d, torch.zeros_like(dot_d))

            dot_s = torch.matmul(reflection, dir)
            dot_s = torch.unsqueeze(dot_s, -1)
            dot_s = torch.maximum(dot_s, torch.zeros_like(dot_s))**shininess

            color = k_a * ambient + k_d * diffuse * dot_d  + k_s * specular * dot_s
            color = torch.clamp(color, max=1.0)

            color = torch.where(D <= dist_threshold, color, torch.ones_like(color))

            pbar.update(40)

        frame_time = time.perf_counter() - start_time

        image_inputs = {"normals" : n, "colors" : color}

        for key in tqdm(image_inputs, f"Saving results into {output_dir}"):
            transposed = torch.transpose(image_inputs[key], 0, 1)

            img = transform(torch.reshape(transposed, (3, n_pixels, n_pixels)))
            img = img.rotate(90)

            file_name = f"{experiment_name}_{key}_LOD_{lod}"
            if not multiscale and not normal_mapping:
                file_name = f"{file_name}_baseline_iters_{num_iterations[lod]}"
            else:
                n_iters = ','.join([str(i) for i in num_iterations])
                file_name = f"{file_name}_multiscale_{multiscale}_mapping_{normal_mapping}_iters_{n_iters}"
            img.save(osp.join(output_dir, file_name + ".png"))

    # Estimate model memory
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = (mem_params + mem_bufs) / 1000 # K-bytes

    return {"name": file_name, "time(s)": frame_time, "memory (KB)": mem}, color


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
        "armadillo_64x1_256x3": osp.join(args.input_folder, './bacon_' + 'armadillo' + '.pth'),
        "buddha_64x1_256x3": osp.join(args.input_folder, './bacon_' + 'buddha' + '.pth'),
        "bunny_64x1_256x3": osp.join(args.input_folder, './bacon_' + 'bunny' + '.pth'),
        "lucy_64x1_256x3": osp.join(args.input_folder, './bacon_' + 'lucy' + '.pth'),
        # "armadillo_256x3": [osp.join(args.input_folder, "siren_armadillo_256x3.pth")],
        # "buddha_64x1_256x3": [osp.join(args.input_folder, "siren_buddha_64x1.pth"),
        #                       osp.join(args.input_folder, "siren_buddha_256x3.pth")],
        # "buddha_64x1_256x2_256x3": [osp.join(args.input_folder, "siren_buddha_64x1.pth"),
        #                             osp.join(args.input_folder, "siren_buddha_256x2.pth"),
        #                             osp.join(args.input_folder, "siren_buddha_256x3.pth")],
        # "buddha_256x3": [osp.join(args.input_folder, "siren_buddha_256x3.pth")],
        # "bunny_64x1_256x1_256x3": [osp.join(args.input_folder, "siren_bunny_64x1.pth"),
        #                            osp.join(args.input_folder, "siren_bunny_256x1.pth"),
        #                            osp.join(args.input_folder, "siren_bunny_256x3.pth")],
        # "bunny_64x1_256x3": [osp.join(args.input_folder, "siren_bunny_64x1.pth"),
        #                      osp.join(args.input_folder, "siren_bunny_256x3.pth")],
        # "bunny_256x3": [osp.join(args.input_folder, "siren_bunny_256x3.pth")],
        # "dragon_64x1_256x1_256x3": [osp.join(args.input_folder, "siren_dragon_64x1.pth"),
        #                             osp.join(args.input_folder, "siren_dragon_256x1.pth"),
        #                             osp.join(args.input_folder, "siren_dragon_256x3.pth")],
        # "dragon_128x1_256x3": [osp.join(args.input_folder, "siren_dragon_128x1.pth"),
        #                        osp.join(args.input_folder, "siren_dragon_256x3.pth")],
        # "dragon_256x3": [osp.join(args.input_folder, "siren_dragon_256x3.pth")],
        # "lucy_64x1_256x3": [osp.join(args.input_folder, "siren_lucy_64x1.pth"),
        #                     osp.join(args.input_folder, "siren_lucy_256x3.pth")],
        # "lucy_64x1_256x2_256x3": [osp.join(args.input_folder, "siren_lucy_64x1.pth"),
        #                           osp.join(args.input_folder, "siren_lucy_256x2.pth"),
        #                           osp.join(args.input_folder, "siren_lucy_256x3.pth")],
        # "lucy_256x3": [osp.join(args.input_folder, "siren_lucy_256x3.pth")],
    }

    max_iters = 100
    level_0_iters = 50

    iterations = {
        "armadillo_64x1_256x3": [[level_0_iters, 30], [level_0_iters, 40], [level_0_iters, 50]],
        "bunny_64x1_256x3": [[level_0_iters, 30], [level_0_iters, 40], [level_0_iters, 50]],
        "buddha_64x1_256x3": [[level_0_iters, 30], [level_0_iters, 40], [level_0_iters, 50]],
        "lucy_64x1_256x3": [[level_0_iters, 30], [level_0_iters, 40], [level_0_iters, 50]],
    }

    deltas = {
        "armadillo_64x1_256x3": [0.00002],
        "bunny_64x1_256x3": [0.00002],
        "buddha_64x1_256x3": [0.00002],
        "lucy_64x1_256x3": [0.00002],
    }

    stats_list = []
    for experiment_name in experiments:
        ckpts = experiments[experiment_name]

        print('\n==========\n' + experiment_name.upper() + '\n==========\n')

        #models = [None]
        # the network has 4 output levels of detail
        max_frequency = 3*(32,)
        #hidden_layers = [2,8]
        hidden_size=256

        # load model
        #models = [None]*2
        bacon_levels =[2,6]
#        for i in range(len(bacon_levels)):
        model = MultiscaleBACON(3, hidden_size, 1,
                                hidden_layers=8,
                                bias=True,
                                frequency=max_frequency,
                                quantization_interval=np.pi,
                                is_sdf=True,
                                output_layers=bacon_levels,
                                reuse_filters=True)
        ckpt = torch.load(ckpts)
        model.load_state_dict(ckpt)
        model.cuda()

        # deltas_calc = calc_deltas(models, device=device)
        # print("DELTAS = ", [d.item() for d in deltas_calc])

        dist_threshold = 4e-2

        print('\n== Bootstraping ==')
        _ = multiscale_sphere_tracing(
            model, experiment_name=experiment_name,
            deltas=deltas[experiment_name], num_iterations=[2] *2,
            n_pixels=args.viewport_size, lod=0,
            multiscale=False, normal_mapping=False,
            dist_threshold=dist_threshold,
            output_dir=args.output_folder,
            device=device
        )

        print('\n== BASELINE Fine ==')
        stats, baseline_color = multiscale_sphere_tracing(
            model, experiment_name=experiment_name,
            deltas=deltas[experiment_name], num_iterations=[max_iters] * 2,
            n_pixels=args.viewport_size, lod=1,
            multiscale=False, normal_mapping=False,
            dist_threshold=dist_threshold,
            output_dir=args.output_folder,
            device=device
        )

        N = args.viewport_size
        baseline_color = baseline_color.reshape(3, N, N).unsqueeze(0)

        stats["psnr"] = psnr(baseline_color, baseline_color, max_val=1.0).item()
        stats["mse"] = ((baseline_color - baseline_color) ** 2).mean().item()
        stats["name"] = "MAX"
        
        stats_list.append(stats)

        for i in range(2):
            print('===== LODS: ' + str(i + 1)  + ' =====')

            if i < (1):
                print('\n== coarse ==')
                stats, colors_coarse = multiscale_sphere_tracing(
                    model, experiment_name=experiment_name,
                    deltas=deltas[experiment_name], num_iterations=[max_iters] * 2,
                    n_pixels=args.viewport_size, lod=i,
                    multiscale=False, normal_mapping=False,
                    dist_threshold=dist_threshold,
                    output_dir=args.output_folder,
                    device=device
                )
                colors_coarse = colors_coarse.reshape(3, N, N).unsqueeze(0)
                stats["psnr"] = psnr(baseline_color, colors_coarse, max_val=1.0).item()
                stats["mse"] = ((baseline_color - colors_coarse) ** 2).mean().item()
                stats_list.append(stats)
            
            print('\n== Baseline few iterations ==')
            baseline_iters = [int(max_iters/2)]
            for j in range(len(baseline_iters)):
                stats, baseline_few_iters = multiscale_sphere_tracing(
                    model, experiment_name=experiment_name, deltas=deltas[experiment_name], num_iterations=[baseline_iters[j]] * 2,
                    n_pixels=args.viewport_size, lod=i,
                    multiscale=False, normal_mapping=False,
                    dist_threshold=dist_threshold,
                    output_dir=args.output_folder,
                    device=device
                )
                    
                baseline_few_iters = baseline_few_iters.reshape(3, N, N).unsqueeze(0)
                stats["psnr"] = psnr(baseline_color, baseline_few_iters.reshape(3, N, N).unsqueeze(0), max_val=1.0).item()
                stats["mse"] = ((baseline_color - baseline_few_iters) ** 2).mean().item()
                stats_list.append(stats)

            if i == 0:
                print("\n== Normal Mapping ==")
                stats, color_nm = multiscale_sphere_tracing(
                    model, experiment_name=experiment_name,
                    deltas=deltas[experiment_name],
                    num_iterations=[max_iters] * 2,
                    n_pixels=args.viewport_size, lod=i,
                    multiscale=False, normal_mapping=True,
                    dist_threshold=dist_threshold,
                    output_dir=args.output_folder,
                    device=device
                )
                color_nm = color_nm.reshape(3, N, N).unsqueeze(0)
                stats["psnr"] = psnr(baseline_color, color_nm.reshape(3, N, N).unsqueeze(0), max_val=1.0).item()
                stats["mse"] = ((baseline_color - color_nm) ** 2).mean().item()
                stats_list.append(stats)

            if i > 0:
                for j in range(len(iterations[experiment_name])):
                    print("\n== Multiscale ==")
                    stats, color_ms = multiscale_sphere_tracing(
                        model, experiment_name=experiment_name,
                        deltas=deltas[experiment_name],
                        num_iterations=iterations[experiment_name][j],
                        n_pixels=args.viewport_size, lod=i,
                        multiscale=True, normal_mapping=False,
                        dist_threshold=dist_threshold,
                        output_dir=args.output_folder,
                        device=device
                    )
                    color_ms = color_ms.reshape(3, N, N).unsqueeze(0)
                    stats["psnr"] = psnr(baseline_color, color_ms.reshape(3, N, N).unsqueeze(0), max_val=1.0).item()
                    stats["mse"] = ((baseline_color - color_ms) ** 2).mean().item()
                    stats_list.append(stats)

            # print("\n== Multiscale, Normal Mapping On ==")
            # stats = multiscale_sphere_tracing(
            #     models, experiment_name=experiment_name, deltas=deltas[experiment_name], num_iterations=iterations[experiment_name],
            #     n_pixels=args.viewport_size, lod=i,
            #     multiscale=True, normal_mapping=True,
            #     dist_threshold=dist_threshold,
            #     output_dir=args.output_folder,
            #     device=device,
            # )
            # stats_list.append(stats)
            # dist_threshold *= 2

    stats_df = pd.DataFrame.from_records(stats_list)
    stats_df.to_csv(osp.join(args.output_folder, "stats.csv"), sep=";",
                    index=None)
