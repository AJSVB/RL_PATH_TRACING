import sys
sys.path.append('..')
import denoising1 as denoising
from training import dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
#import time.training.dataset as dataset
path = "/home/ascardigli/blender-3.2.2-linux-x64/suntemple/"

from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import mean_squared_error
psnr = PeakSignalNoiseRatio()#.cuda(0)

spp=8


a=dataset.ValidationDataset(None, "nul")
import torch
pid="0"
m_denoised = []
m_ray= []

p_denoised = []
p_ray= []
i=10
import  math

def f(spp,i):
 color = a.data(i).cpu()*math.sqrt(spp+1) # (5/8+3*spp/8)
 temp = dataset.get_aux(a.path,i)
 gd = torch.Tensor(dataset.get_truth(a.path,i))
 albedo,normal  = temp[:,:,:3],temp[:,:,3:6]
 denoising.tensor_to_pfm(normal,"normal")
 denoising.tensor_to_pfm(albedo,"albedo")
 color = color[:spp+(spp==0)].mean(0).permute(1,2,0)
 print(torch.max(color))
 ray = color
 color=denoising.tensor_to_pfm(color,"color")
 denoising.denoiser("color","albedo","normal",pid)
 b=torch.Tensor(denoising.pfm_to_tensor("/home/ascardigli/RL_PATH_TRACING/tmp/"+pid))
 m_denoised.append(mean_squared_error(b,gd))
 p_denoised.append(psnr(b,gd))
 m_ray.append(mean_squared_error(ray,gd))
 p_ray.append(psnr(ray,gd))
 return m_denoised,p_denoised,m_ray,p_ray




for spp in range(0,9):
 for i in range(800,900):
  m_denoised,p_denoised,m_ray,p_ray=f(spp,i)
 with open("comp/"+str(spp)+'msesdenoised.txt', 'w') as fp:
        fp.write("\n")
 with open("comp/"+str(spp)+'psnrsdenoised.txt', 'w') as fp:
        fp.write("\n")
 with open("comp/"+str(spp)+'msesray.txt', 'w') as fp:
        fp.write("\n")
 with open("comp/"+str(spp)+'psnrsray.txt', 'w') as fp:
        fp.write("\n")


 with open("comp/"+str(spp)+'msesdenoised.txt', 'a') as fp:
         fp.write("\n".join(str(item.item()) for item in m_denoised))
         fp.write("\n")
 with open("comp/"+str(spp)+'psnrsdenoised.txt', 'a') as fp:
         fp.write("\n".join(str(item.item()) for item in p_denoised))
         fp.write("\n")
 with open("comp/"+str(spp)+'msesray.txt', 'a') as fp:
         fp.write("\n".join(str(item.item()) for item in m_ray))
         fp.write("\n")
 with open("comp/"+str(spp)+'psnrsray.txt', 'a') as fp:
         fp.write("\n".join(str(item.item()) for item in p_ray))
         fp.write("\n")



plt.imshow(gd)
plt.savefig("images/target.png")
plt.clf()
plt.imshow(b)
plt.savefig("images/b.png")
plt.clf()
plt.imshow(ray)
plt.savefig("images/ray.png")
plt.clf()

