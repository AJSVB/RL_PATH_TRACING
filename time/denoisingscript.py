import sys
import denoising as denoising
from training import dataset
import numpy as np
import torch
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import mean_squared_error
sys.path.append("..")

psnr = PeakSignalNoiseRatio().cuda(0)

"""
This script computes the mse and psnrs for vanilla ray tracing and OIDN-denoised ray tracing, given a path and number interval for test data
"""

a = dataset.Dataset()
pid = "0"


def f(spp, i, m_denoised, p_denoised):
    """denoises one frame

    Args:
        spp (int): the spp count
        i (int): the feame number
        m_denoised (list): list of mse for denoised frames, changed by reference
        p_denoised (list): list of psnr for denoised frames, changed by reference
    """
    color = (
        a.data(i).cpu() 
    )  
    temp1, gd = a.get(i)
    temp1 = torch.Tensor(temp1).cuda().permute(2, 0, 1)
    normal, albedo, depth = temp1[:3].cpu() / 255, temp1[3:6].cpu() / 255, temp1[6:]
    gd = torch.Tensor(gd).cuda().permute(2, 0, 1)  

    normal = normal.permute(1, 2, 0)
    albedo = albedo.permute(1, 2, 0)
    color = color[:spp].mean(0)
    color = color.permute(1, 2, 0)

    denoising.tensor_to_pfm(normal, "normal")
    denoising.tensor_to_pfm(albedo, "albedo")
    denoising.tensor_to_pfm(color, "color")
    denoising.denoiser("color", "albedo", "normal", pid)
    b = (
        torch.Tensor(
            denoising.pfm_to_tensor("/home/ascardigli/RL_PATH_TRACING/tmp/" + pid)
        )
        .cuda(0)
        .permute(2, 0, 1)
    )
    m_denoised.append(mean_squared_error(b, gd))
    p_denoised.append(psnr(b, gd).cpu())


for spp in [1, 4]: 
    m_denoised = []
    p_denoised = []
    for i in range(3300, 3400):
        f(spp, i, m_denoised, p_denoised)

    with open("comp/" + str(spp) + "msesdenoised.txt", "w") as fp:
        fp.write("\n")
    with open("comp/" + str(spp) + "psnrsdenoised.txt", "w") as fp:
        fp.write("\n")

    with open("comp/" + str(spp) + "msesdenoised.txt", "a") as fp:
        fp.write("\n".join(str(item.item()) for item in m_denoised))
        fp.write("\n")
    with open("comp/" + str(spp) + "psnrsdenoised.txt", "a") as fp:
        fp.write("\n".join(str(item.item()) for item in p_denoised))
        fp.write("\n")
