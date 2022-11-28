import os
from glob import glob
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler



from .image import *

class PreprocessedDataset(Dataset):
  def __init__(self, cfg, name):
    super(PreprocessedDataset, self).__init__()
    self.tile_size = cfg.tile_size
    if not name:
      self.samples = []
      self.num_images = 0
      return


## -----------------------------------------------------------------------------
## Training dataset
## -----------------------------------------------------------------------------
from PIL import Image
import torchvision.transforms.functional as TF


import functools
@functools.lru_cache(maxsize=1000)
def get_ith_image(path,i,frame_number):
    image = Image.open(path+str(i).zfill(2) + "-" + str(frame_number).zfill(4)+'.png')
    #x = TF.to_tensor(image)
    #x.unsqueeze_(0)
    return np.expand_dims(image,0)[:,:720,:720]/255.


@functools.lru_cache(maxsize=1000)
def get_truth(path,frame_number):
    image= Image.open(path + "gd"+str(frame_number).zfill(4)+".png")
    #x = TF.to_tensor(image)
    return np.array(image)[:720,:720]/255.

def get_add(a,b,c):
    if b=="UVUV":
      b="00UVUV"
    image= Image.open(a+b+c)
    z= np.array(image)[:720,:720]
#    temp = [np.sum(z[:,:,i]) for i in range(3)]
#    if not (temp-temp[0]).any():
#      z = z[:,:,0:1]
    b=np.min(z)
    a=np.max(z)
    if b==a:
      a = a if a else 1.
      return z/a
    else:
      return (z-b)/(a-b)*1.

@functools.lru_cache(maxsize=1000)
def get_aux(path,frame_number):
    f=path + "add"
    end = str(frame_number).zfill(4)+".png"
    imgs = np.concatenate([get_add(f,2*g,end) for g   in \
# ["Alpha","Vector","IndexOB","IndexMA", "DiffCol",\
["Denoising Normal","Denoising Albedo","Denoising Depth"] # \
#,"UV"\
],-1)
    return imgs


#get_aux.cache_clear()
#get_truth.cache_clear()
#get_ith_image.cache_clear()
#import gc
#gc.collect()
#objects = [i for i in gc.get_objects() if isinstance(i, functools._lru_cache_wrapper)]
#for object in objects:
# object.cache_clear()
#import matplotlib.pyplot as plt

class ValidationDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
#    super(ValidationDataset, self).__init__(cfg, name)

    self.path = "/home/ascardigli/blender-3.2.2-linux-x64/suntemple/"
    self.temp = [np.load("temp"+str(i)+".npy") for i in range(14)]
    sampling = torch.arange(1,9).reshape(8,1,1,1).cuda(0)
    self.sampling = sampling.repeat(1,1,720,720)
    self.num_images=1100

     
  def __len__(self):
    return self.num_images


  def data(self,index):
    samples = torch.cat([torch.Tensor(get_ith_image(self.path,i,index)).cuda(0) for i in range(8)],0)
    return samples.permute(0,3,1,2)


  def translation(self,i,data):
   data = data.reshape(-1,720,720)
   a=(i)//100 #TODO check
   b=(i)%100
   warp_matrix = self.temp[a][b] 
   flow = torch.nn.functional.affine_grid(torch.Tensor(warp_matrix).unsqueeze(0).cuda(0),\
(1,3,720,720), align_corners=True)
   temp= torch.nn.functional.grid_sample(.1+data.unsqueeze(0),\
flow,align_corners=True) 
   temp[temp==0]=-.9 #TODO
   return (temp-.1).squeeze(0)


  def generate(self,samples,idxs,i):
    samples = torch.cat([samples,-1*torch.ones((1,3,720,720)).cuda(0)],0)
    idxs= idxs.reshape(1,720,720) 
    sampling = idxs - self.sampling 
    sampling[sampling<0] = 8
    """
    def p(a,b):
      print( a+" "+str(b.item()))
    p("0",torch.sum(sampling==0))
    p("1",torch.sum(sampling==1))
    p("2",torch.sum(sampling==2))
    p("3",torch.sum(sampling==3))
    p("4",torch.sum(sampling==4))
    p("5",torch.sum(sampling==5))
    p("6",torch.sum(sampling==6))
    p("7",torch.sum(sampling==7))
    p("8",torch.sum(sampling==8))
    """
    sampling = sampling.repeat(1,3,1,1)
    temp = torch.take_along_dim(samples,sampling,dim=0)

    return temp


  def get(self, index):
    target_image = get_truth(self.path,index)
    input_image = get_aux(self.path,index)
    input_image=input_image.reshape(*input_image.shape[:2],-1)
    return input_image, target_image


