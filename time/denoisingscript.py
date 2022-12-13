import sys
sys.path.append('..')
import denoising1 as denoising
from training import dataset
import numpy as np
import matplotlib.pyplot as plt
#import time.training.dataset as dataset
path = "/home/ascardigli/blender-3.2.2-linux-x64/suntemple/"

a=dataset.ValidationDataset(None, "nul")
import torch
pid="0"
for i in range(100):
 print(i)
 color = a.data(i).cpu()#*255
 #print(torch.max(color))
 temp = dataset.get_aux(a.path,i)#*255
 albedo,normal  = temp[:,:,:3],temp[:,:,3:6]
 #print(np.max(normal)) 
 denoising.tensor_to_pfm(normal,"normal")
 denoising.tensor_to_pfm(albedo,"albedo")
 color = color[:4].mean(0).permute(1,2,0)
 color=denoising.tensor_to_pfm(color,"color")
 denoising.denoiser("color","albedo","normal",pid)
 b=denoising.pfm_to_tensor("/home/ascardigli/RL_PATH_TRACING/tmp/"+pid)
 print(np.max(b))
 plt.imshow(b*5)
 plt.savefig("images/"+str(i)+".png")

