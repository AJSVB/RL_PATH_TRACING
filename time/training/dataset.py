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
@functools.cache
def get_ith_image(path,i,frame_number):
    image = Image.open(path+str(i).zfill(2) + "-" + str(frame_number).zfill(4)+'.png')
    #x = TF.to_tensor(image)
    #x.unsqueeze_(0)
    return np.expand_dims(image,0)[:,:720,:720]/255.


@functools.cache
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

@functools.cache
def get_aux(path,frame_number):
    f=path + "add"
    end = str(frame_number).zfill(4)+".png"
    imgs = np.concatenate([get_add(f,2*g,end) for g   in \
 ["Image","Alpha","Depth","Mist","Position","Normal","Vector","IndexOB","IndexMA",\
    "DiffCol","Denoising Normal","Denoising Albedo","Denoising Depth","UV"]],-1)
    return imgs




import matplotlib.pyplot as plt

class ValidationDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
    super(ValidationDataset, self).__init__(cfg, name)

    self.path = "/home/ascardigli/blender-3.2.2-linux-x64/suntemple/"
    self.temp = [np.load("temp"+str(i)+".npy") for i in range(14)]
    sampling = np.arange(8).reshape(8,1,1,1)
    sampling = np.repeat(sampling,3,axis=1) 
    sampling = np.repeat(sampling,720,axis=2)
    self.sampling = np.repeat(sampling,720,axis=3) 
    self.num_images=400

     
  def __len__(self):
    return self.num_images


  def data(self,index):
    samples = np.concatenate([get_ith_image(self.path,i,index) for i in range(8)],0)
    return samples.transpose(0,3,1,2)


  def translation(self,i,data):
   a=(i-1)//100
   b=(i-1)%100
   warp_matrix = self.temp[a][b] 
   flow = torch.nn.functional.affine_grid(torch.Tensor(warp_matrix).unsqueeze(0)[:,::,:],(1,3,720,720), align_corners=True).repeat(8,1,1,1)
   temp= torch.nn.functional.grid_sample(.1+data,\
flow,align_corners=True) 
   temp[temp==0]=-.9 #TODO
   return temp-.1


  def generate(self,samples,idxs,old_data,i):
    if i !=0:
     old_data = self.translation(i,old_data)
    samples = np.concatenate([samples,old_data],0)
    idxs=np.repeat(idxs.reshape(1,720,720),3,axis=0)
    sampling = self.sampling
    sampling = sampling - idxs

    temp = torch.Tensor(np.take_along_axis(samples,sampling,0))

    import torchvision.transforms as T

    for j in range(8):
     img= T.ToPILImage()(temp[j])
     img.save(str("images/"+str(i)+"_"+str(j)+".png"))

    return temp


  def get(self, index):
    target_image = get_truth(self.path,index)
    input_image = get_aux(self.path,index)
    input_image=input_image.reshape(*input_image.shape[:2],-1)
    return input_image, target_image


