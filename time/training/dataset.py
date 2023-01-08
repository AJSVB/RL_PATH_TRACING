import os
from glob import glob
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

#import matplotlib.pyplot as plt


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

@functools.lru_cache(maxsize=1000)
def get_flow(path,frame_number):
  a=path+str(frame_number).zfill(4)+"corr.pt"
  return torch.load(a)[:,:720,:720]


class ValidationDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
#    super(ValidationDataset, self).__init__(cfg, name)

    self.path = "/home/ascardigli/blender-3.2.2-linux-x64/suntemple/"
    self.path2 ="/home/ascardigli/blender-3.2.2-linux-x64/zeroday/"
    sampling = torch.arange(1,9).reshape(8,1,1,1).cuda(0)
    self.sampling = sampling.repeat(1,1,720,720)
    self.num_images=1600
    self.bb = torch.Tensor(np.tile(np.arange(720),(720,1))).cuda(0)
    self.aa = torch.Tensor(np.tile(np.arange(720).T,(720,1)).T).cuda(0)


     
  def __len__(self):
    return self.num_images


  def data(self,index):
    samples = torch.cat([torch.Tensor(get_ith_image(self.get_path(index),i,self.get_i(index))).cuda(0) for i in range(8)],0)
    return samples.permute(0,3,1,2)[torch.randperm(8)]

  def get_path(self,i):
   if i<1200: 
    return self.path
   return self.path2
   

  def get_i(self,i):
   if i <1200:
    return i

   return i-1200

  def translation(self,i,data,transform=None):
   data = data.reshape(-1,720,720)
   flow = get_flow(self.get_path(i)+"motions/",self.get_i(i+1)).cuda(0) #flow from i to i+1
   

   a=720
   b=720


   flow[:,:,:,0] = flow[:,:,:,0]  - 2*self.bb/b
   flow[:,:,:,1] = flow[:,:,:,1]  - 2*self.aa/a	

   flow[:,:,:,0] = transform(flow[:,:,:,0])
   flow[:,:,:,1] = transform(flow[:,:,:,1])

   flow[:,:,:,0] = flow[:,:,:,0]  + 2*self.bb/b
   flow[:,:,:,1] = flow[:,:,:,1]  + 2*self.aa/a


#   temp= torch.nn.functional.grid_sample(data.unsqueeze(0),flow) 
#   return temp.squeeze(0)
   temp= torch.nn.functional.grid_sample(.1+data.unsqueeze(0),flow,align_corners=False) 
   temp[temp==0]=-.9 #TODO
   return (temp-.1).squeeze(0)


  def generate(self,samples,idxs,i):
    samples = samples[torch.randperm(8)]
    samples = torch.cat([samples,-1*torch.ones((1,3,720,720)).cuda(0)],0)
    idxs= idxs.reshape(1,720,720) 
    sampling = idxs - self.sampling 
    sampling[sampling<0] = 8
    sampling = sampling.repeat(1,3,1,1)
    temp = torch.take_along_dim(samples,sampling,dim=0)

    return temp


  def get(self, index):
    target_image = get_truth(self.get_path(index),self.get_i(index))
    input_image = get_aux(self.get_path(index),self.get_i(index))
    input_image=input_image.reshape(*input_image.shape[:2],-1)
    return input_image, target_image


