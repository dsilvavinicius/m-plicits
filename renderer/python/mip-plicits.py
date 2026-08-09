import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import modules
import torchvision.transforms as T
from PIL import Image
import time

def multiscale_sphere_tracing(ckpt_path, name, lod, dist_threshold, multiscale=True, normal_mapping = True, hidden_layers=8, hidden_size=256,
output_layers=[1, 2, 4, 8]):
    time.perf_counter()
    
    pos_cpu = torch.zeros(N * N, 3, requires_grad=True)
    
    with torch.no_grad():
        # rays initialization
        for i in tqdm(range(0, N), 'Rays initialization'):
            for j in range(0, N):
                # Orthogonal projection using Normalized Device Coordinates.
                if name == 'lucy':
                    pos_cpu[i * N + j, 0] = i / N - 0.5 
                    pos_cpu[i * N + j, 1] = 0.6
                    pos_cpu[i * N + j, 2] = j / N - 0.5
                else:
                    pos_cpu[i * N + j, 0] = i / N - 0.5 
                    pos_cpu[i * N + j, 1] = j / N - 0.5
                    
                    if name == 'bunny' or name == 'buddha':
                        pos_cpu[i * N + j, 2] = 0.6
                    else:
                        pos_cpu[i * N + j, 2] = -0.6

        pos_gpu = pos_cpu.cuda()
        
        if name == 'lucy':
            dir = torch.tensor([0.0, -1.0, 0.0]).cuda()
        elif name == 'bunny' or name == 'buddha':
            dir = torch.tensor([0.0, 0.0, -1.0]).cuda()
        else:
            dir = torch.tensor([0.0, 0.0, 1.0]).cuda()

        distances = torch.zeros(N * N, 1).cuda()
        transform = T.ToPILImage()

        start_time = time.perf_counter()

        # the network has 4 output levels of detail
        max_frequency = 3*(32,)

        # load model
        model = modules.MultiscaleBACON(3, hidden_size, 1,
                                        hidden_layers=hidden_layers,
                                        bias=True,
                                        frequency=max_frequency,
                                        quantization_interval=np.pi,
                                        is_sdf=True,
                                        output_layers=output_layers,
                                        reuse_filters=True)

        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)
        model.cuda()

        # multiscale sphere tracing loop
        if multiscale:
            lod_init = 0
        else:
            lod_init = lod

        for i in range(lod_init, lod + 1):
            for j in tqdm(range(0, 50), 'Sphere tracing - LOD ' + str(i)):
                # infer distance
                out = model({'coords': pos_gpu})['model_out']
                if i == lod:
                    distances = out[i]
                else:
                    distances = out[i] - lod_deltas[i]

                # update pos
                pos_gpu = pos_gpu + torch.mul(distances, dir)
            
            img = transform(torch.reshape(distances, (1, N, N)))
            img = img.rotate(90)
            img.save('results/' + name + '_lod_' + str(i) + '.png')

    D = torch.cat((distances, distances, distances), dim=-1)

    result = model({'coords': pos_gpu})
    coords = result['model_in']

    if normal_mapping:
        distances = result['model_out'][n_lods - 1]
        pbar_str = 'Neural Implicits Normal Mapping - Mapping LOD ' + str(n_lods - 1) + ' onto LOD ' + str(lod)
    else:
        distances = result['model_out'][lod]
        pbar_str = 'Normals - LOD ' + str(lod)
    # Normals.
    with tqdm(total=100, desc=pbar_str) as pbar:
        grad = torch.autograd.grad(distances, coords, grad_outputs=torch.ones_like(distances))[0]
        
        pbar.update(70)

        with torch.no_grad():
            # Normalization and distance condition
            grad_norm = torch.linalg.norm(grad, dim=-1)

            unit_grad = grad/grad_norm.unsqueeze(-1)
            unit_grad = torch.abs(unit_grad)

            unit_grad = torch.where(D<=dist_threshold, unit_grad, torch.ones_like(unit_grad))

            pbar.update(30)

    # Rendering
    with torch.no_grad():
        with tqdm(total=100, desc='Rendering') as pbar:
            k_s = 0.5
            k_d = 1.0
            k_a = 1.0
            shininess = 35.0

            ambient = torch.tensor([0.2, 0.2, 0.2]).cuda()
            specular = torch.tensor([1.0, 1.0, 1.0]).cuda()
            diffuse = torch.tensor([0.54, 0.54, 0.54]).cuda()

            if name == 'lucy':
                light_dir = torch.tensor([0.0, 1.0, 0.0]).cuda()
            else:
                light_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
            
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

            color = torch.where(D<=dist_threshold, color, torch.ones_like(color))

            pbar.update(40)
    
        frame_time = time.perf_counter() - start_time 

        image_inputs = {'normals' : n, 'colors' : color}

        for key in tqdm(image_inputs, 'Saving results into ./results'):
            transposed = torch.transpose(image_inputs[key], 0, 1)
            
            img = transform(torch.reshape(transposed, (3, N, N)))
            img = img.rotate(90)

            if multiscale is False and normal_mapping is False:
                file_name = '%s_%s_LOD_%d_baseline' % (name, key, lod)
            else:
                file_name = '%s_%s_LOD_%d_multiscale_%s_mapping_%s' % (name, key, lod, multiscale, normal_mapping)
            img.save('results/' + file_name + '.png')
        
        # Estimate model memory
        mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
        mem = (mem_params + mem_bufs) / 1000 # K-bytes

        return {'name': file_name, 'time(s)' : frame_time, 'memory (KB)' : mem}

if __name__ == '__main__':
    global N, output_layers, dist_threshold, max_level, lod_deltas
    N = 512
    output_layers = [2, 4, 6, 8]
    num_outputs = len(output_layers)
    n_lods = 4
    lod_deltas = [8.0e-3, 4.0e-3, 2.0e-3, 1.0e-3]

    experiments = ['armadillo']#['bunny', 'dragon', 'armadillo', 'lucy', 'thai']
    stats_list = []

    for experiment in experiments:
        ckpt = './data/armadillo_64x1' + '.pth'

        print('\n=================\n' + experiment.upper() + '\n=================\n')

        for i in range(0, n_lods):

            print('===== LODS: ' + str(i + 1)  + ' =====')

            dist_thresholds = (1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3)

            print('\n== Baseline ==')

            stats = multiscale_sphere_tracing(ckpt, name=experiment, lod=i, dist_threshold=dist_thresholds[i], multiscale=False,
            normal_mapping=False, output_layers=output_layers)
            stats_list.append(stats)

            if i > 0:
                print('\n== Multiscale, Normal Mapping Off ==')

                stats = multiscale_sphere_tracing(ckpt, name=experiment, lod=i, dist_threshold=dist_thresholds[i], multiscale=True,
                normal_mapping=False, output_layers=output_layers)
                stats_list.append(stats)

            print('\n== Multiscale, Normal Mapping On ==')

            stats = multiscale_sphere_tracing(ckpt, name=experiment, lod=i, dist_threshold=dist_thresholds[i], multiscale=True,
            normal_mapping=True, output_layers=output_layers)
            stats_list.append(stats)

            print('\n')

    stats_df = pd.DataFrame.from_records(stats_list)
    stats_df.to_csv('results/stats.csv')