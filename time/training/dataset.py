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
#@functools.cache
def get_ith_image(path,i,frame_number):
    image = Image.open(path+str(i).zfill(2) + "-" + str(frame_number).zfill(4)+'.png')
    #x = TF.to_tensor(image)
    #x.unsqueeze_(0)
    return np.expand_dims(image,0)[:,:720,:720]/255.


#@functools.cache
def get_truth(path,frame_number):
    image= Image.open(path + "gd"+str(frame_number).zfill(4)+".png")
    #x = TF.to_tensor(image)
    return np.array(image)[:720,:720]/255.

#@functools.cache
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


  def translation(self,i,data):
   img1 = get_truth(self.path,i-1)
   img2 = get_truth(self.path,i)
   import cv2
   warp_matrix = np.eye(2, 3, dtype=np.float32)
   warp_mode = cv2.MOTION_AFFINE
   termination_eps = 1e-5

   im1_gray = cv2.cvtColor(np.float32(img1),cv2.COLOR_BGR2GRAY)
   im2_gray = cv2.cvtColor(np.float32(img2),cv2.COLOR_BGR2GRAY)
   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50000,  termination_eps)
   (cc, warp_matrix) = cv2.findTransformECC(im2_gray,im1_gray,warp_matrix, warp_mode, criteria)
   warp_matrix[:,2] = warp_matrix[:,2]/((2*720,2*1280)) 
   flow = torch.nn.functional.affine_grid(torch.Tensor(warp_matrix).unsqueeze(0)[:,::,:],(1,3,720,1280), align_corners=True).repeat(8,1,1,1)
   return torch.nn.functional.grid_sample(data.permute(3,2,0,1),\
flow,align_corners=True).permute(2,3,1,0) 


  def generate(self,samples,idxs,old_data,i):
    old_data = old_data.reshape(720,720,3,8)
    #old_data = self.translation(i,old_data)
#    old_data[old_data==0]=-1 #TODO
    print(old_data[:,:,:,0].sum())
    print(old_data[:,:,:,1].sum())
    print(old_data[:,:,:,2].sum())
    print(old_data[:,:,:,3].sum())
    print(old_data[:,:,:,4].sum())
    print(old_data[:,:,:,5].sum())
    print(old_data[:,:,:,6].sum())
    print(old_data[:,:,:,7].sum())
    samples = np.transpose(samples.reshape(720,720,3,8),(3,0,1,2))
    samples = np.concatenate([samples,np.ones((1,720,720,3))*-1],0)
    idxs=np.repeat(idxs.reshape(720,720,1),3,axis=-1)
    sampling = np.arange(8).reshape(8,1,1,1)
    sampling = np.repeat(sampling,720,axis=1)
    sampling = np.repeat(sampling,720,axis=2) 
    sampling = np.repeat(sampling,3,axis=3) 
    sampling = sampling - idxs
    sampling[sampling<0] = 8
    print(np.sum(sampling==0))
    print(np.sum(sampling==1))
    print(np.sum(sampling==2))
    print(np.sum(sampling==3))
    print(np.sum(sampling==4))
    print(np.sum(sampling==5))
    print(np.sum(sampling==6))
    print(np.sum(sampling==7))
    print(np.sum(sampling==8))

    temp = torch.Tensor(np.take_along_axis(samples,sampling,0))
    old_data=temp
    print(old_data[0].sum())
    print(old_data[1].sum())
    print(old_data[2].sum())
    print(old_data[3].sum())
    print(old_data[4].sum())
    print(old_data[5].sum())
    print(old_data[6].sum())
    print(old_data[7].sum())

    print()
    return temp
  def get(self, index):
    sy = sx = self.tile_size

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
    aux = get_aux(self.path,index)
    input_image=input_image.reshape(*input_image.shape[:2],-1)
    input_image=np.concatenate([input_image,aux],-1)
    input_image=input_image.reshape(*input_image.shape[:2],-1)
    input_image  = input_image [:720,:720]
    temp= image_to_tensor(input_image.copy()).unsqueeze(0).float()
    return temp
