import simulation
import denoising
import torch
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def MSE(a,b):
    return np.sum((a.numpy()-b.numpy())**2)
def SSIM(a,b):
    return ssim(torch.Tensor(a),b,data_range=1).item()
def MultiSSIM(a,b):
    return ms_ssim(torch.Tensor(a),torch.Tensor(b),data_range=1).item()

import random

dataset = simulation.aggregate_by_pixel("../datasets/temple/",100,HEIGHT=720,WIDTH=1280)
gd= simulation.get_truth("../datasets/temple/"+"truth.png",720,1280)


def MSE(a,b):
    return np.sum((a.numpy()-b.numpy())**2)
def SSIM(a,b):
    return ssim(torch.Tensor(a),b,data_range=1).item()
def MultiSSIM(a,b):
    return ms_ssim(torch.Tensor(a),torch.Tensor(b),data_range=1).item()



def average(dataset,number,a):
    sampling = (torch.rand(dataset.shape[3])*dataset.shape[3]).long()[:number]
    temp = dataset[:,:,:,sampling]
    if number == 0:
        temp = torch.zeros(dataset[:,:,:,0].squeeze(-1).shape)
    elif number ==1:
        temp = temp.squeeze(-1)
    elif number != 1:
        temp = temp.mean(dim = 3)
    if a=="denoised":
      temp = torch.Tensor(denoising.denoise(temp,str(0)))
    return temp

def f(i,a):
    return average(dataset,i,a)

truth = gd

for a in ["denoised","notdenoised"]:
 mse_a = []
 ssim_a = []
 mssim_a = []
 for i in range(0,100):
    temp = f(i,a).permute([2,0,1]).unsqueeze(0)
    mse_a.append(MSE(truth,temp))
    ssim_a.append(SSIM(truth,temp))
    mssim_a.append(MultiSSIM(truth,temp))

 print(a+"mse"+str(mse_a))
 print(a+"ssim"+str(ssim_a))
 print(a+"msssim"+str(mssim_a))
