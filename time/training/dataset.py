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
    return np.array(image)/255.

@functools.cache
def get_add(a,b,c):
    if b=="UVUV":
      b="00UVUV"
    image= Image.open(a+b+c)
    z= np.expand_dims(image,-1)
    b=np.min(z)
    a=np.max(z)
    if b==a:
      a = a if a else 1.
      return z/a
    else:
      return (z-b)/(a-b)*1.
    return np.expand_dims(image,-1)


def get_aux(path,frame_number):
    f=path + "add"
    end = str(frame_number).zfill(4)+".png"
    imgs = np.concatenate([get_add(f,2*g,end) for g   in \
 ["Image","Alpha","Depth","Mist","Position","Normal","Vector","IndexOB","IndexMA",\
    "DiffCol","Denoising Normal","Denoising Albedo","Denoising Depth","UV"]],-1)
    return imgs





class ValidationDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
    super(ValidationDataset, self).__init__(cfg, name)

    self.path = "/home/ascardigli/blender-3.2.2-linux-x64/suntemple/"

    self.num_images=400
    self.tiles = []

    if self.num_images == 0:
      return

     
  def __len__(self):
    return self.num_images


  def data(self,index):
    samples = np.concatenate([get_ith_image(self.path,i,index) for i in range(8)],0)
    return samples

  def generate(self,samples,idxs):
    samples = np.transpose(samples.reshape(720,720,3,8),(3,0,1,2))
    samples = np.concatenate([samples,np.ones((1,720,720,3))*-1],0)
    idxs=np.repeat(idxs.reshape(720,720,1),3,axis=-1)
    idxs[idxs<0]=0
    idxs[idxs>7] = 7
    sampling = np.arange(8).reshape(8,1,1,1)
    sampling = np.repeat(sampling,720,axis=1)
    sampling = np.repeat(sampling,720,axis=2) 
    sampling = np.repeat(sampling,3,axis=3) 
    sampling = sampling - idxs
    sampling[sampling<0] = 8
    temp = torch.Tensor(np.take_along_axis(samples,sampling,0))
    return temp
  def get(self, index):
    sy = sx = self.tile_size

    input_name = "-"+str(index).zfill(4)+".png"
    target_name = "gd"+str(index).zfill(4)+".png"
    
    target_image = get_truth(self.path,index)
    input_image = get_aux(self.path,index)
    input_image=input_image.reshape(*input_image.shape[:2],-1)
    input_image  = input_image [:720,:720]
    target_image = target_image[:720,:720]
    return image_to_tensor(input_image.copy()), image_to_tensor(target_image.copy())


  def __getitem__(self, index,samples=None):
    a=time.time()
    if samples is None:  
      sample = self.sample(index) 
    input_image=samples
    aux = get_aux(self.path,index)[:720,:720]
    input_image=np.concatenate([input_image,aux],-1)
    input_image=input_image.reshape(*input_image.shape[:2],-1)
    input_image  = input_image [:720,:720]
    temp= image_to_tensor(input_image.copy()).unsqueeze(0).float()
    return temp
