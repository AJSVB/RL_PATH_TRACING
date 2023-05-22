import sys
from training import dataset
import numpy as np
import torch
import torchvision.transforms as T
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import mean_squared_error
psnr = PeakSignalNoiseRatio().cuda(0)
sys.path.append("..")

a = dataset.Dataset()


def f(spp, i, m_ray, p_ray):
    """Main function for the path tracing algorithm

    Args:
        spp (int): the spp budget
        i (int): the frame number
        m_ray (list): list of logged mse, changed by reference
        p_ray (list): list of logged psnr, changed by reference

    """    
    color = a.data(i)[:spp].cuda().mean(0)
    temp1, gd = a.get(i)
    gd = torch.Tensor(gd).permute(2, 0, 1).cuda()
    temp1 = torch.Tensor(temp1).cuda()
    m_ray.append(mean_squared_error(color, gd))
    p_ray.append(psnr(color, gd).cpu())
    print(p_ray[-1])
    save(color, "images/" + str(spp) + "svgf" + str(i) + ".png")


for spp in [1, 2, 4]:
    m_ray = []
    p_ray = []
    for i in range(700, 800):
        f(spp, i, m_ray, p_ray)
    with open("comp/" + str(spp) + "msespt.txt", "w") as fp:
        fp.write("\n")
    with open("comp/" + str(spp) + "psnrspt.txt", "w") as fp:
        fp.write("\n")
    with open("comp/" + str(spp) + "msespt.txt", "a") as fp:
        fp.write("\n".join(str(item.item()) for item in m_ray))
        fp.write("\n")
    with open("comp/" + str(spp) + "psnrspt.txt", "a") as fp:
        fp.write("\n".join(str(item.item()) for item in p_ray))
        fp.write("\n")
    print(np.mean(p_ray))
