#!/usr/bin/env python
# coding: utf-8


from PIL import Image
from kornia.metrics import psnr
import torch
from torchvision.transforms import ToTensor, ToPILImage


t = ToTensor()
im = ToPILImage()

# The ranges 10:1060 and 480:1445 were found empirically to fit the
# armadillo image in the tightest possible manner.

baseline_im = Image.open("data/neural_vs_traditional_normal_mapping/ground_truth.png")
#baseline_t = t(baseline_im)[:3, 10:1060, 480:1445].unsqueeze(0)
baseline_t = t(baseline_im).unsqueeze(0)

neural_im = Image.open("data/neural_vs_traditional_normal_mapping/neural_normal_mapping.png")
#neural_t = t(neural_im)[:3, 10:1060, 480:1445].unsqueeze(0)
neural_t = t(neural_im).unsqueeze(0)

traditional_im = Image.open("data/neural_vs_traditional_normal_mapping/traditional_normal_mapping.png")
#traditional_t = t(traditional_im)[:3, 10:1060, 480:1445].unsqueeze(0)
traditional_t = t(traditional_im).unsqueeze(0)

print("========== BASELINE VS TRADITIONAL NORMAL MAPPING ==========")
print("PSNR: ", psnr(baseline_t, traditional_t, max_val=1.0).item())
print("MSE: ", ((baseline_t - traditional_t) ** 2).mean().item())

print("========== BASELINE VS NEURAL NORMAL MAPPING ==========")
print("PSNR: ", psnr(baseline_t, neural_t, max_val=1.0).item())
print("MSE: ", ((baseline_t - neural_t) ** 2).mean().item())
