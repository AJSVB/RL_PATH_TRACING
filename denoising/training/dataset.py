import os
from glob import glob
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import *
from util import *
from image import *
from color import *
import tza


def get_data_loader(rank, cfg, dataset, shuffle=False):
  if cfg.num_devices > 1:
    sampler = DistributedSampler(dataset,
                                 num_replicas=cfg.num_devices,
                                 rank=rank,
                                 shuffle=shuffle)
  else:
    sampler = None

  loader = DataLoader(dataset,
                      batch_size=(cfg.batch_size // cfg.num_devices),
                      sampler=sampler,
                      shuffle=(shuffle if sampler is None else False),
                      num_workers=cfg.num_loaders,
                      pin_memory=(cfg.device != 'cpu'))

  return loader, sampler


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
def get_ith_image(path,i,frame_number):
    image = Image.open(path+str(i).zfill(2) + "-" + str(frame_number).zfill(4)+'.png')
    #x = TF.to_tensor(image)
    #x.unsqueeze_(0)
    return np.expand_dims(image,0)

def get_truth(path,frame_number):
    image= Image.open(path + "gd"+str(frame_number).zfill(4)+".png")
    #x = TF.to_tensor(image)
    return np.array(image)


def get_add(a,b,c):
    if b=="UVUV":
      b="00UVUV"
    image= Image.open(a+b+c)
   # x = TF.to_tensor(image)
    return np.expand_dims(image,-1)


def get_aux(path,frame_number):
    f=path + "add"
    end = str(frame_number).zfill(4)+".png"
    imgs = np.concatenate([get_add(f,2*g,end) for g   in \
 ["Image","Alpha","Depth","Mist","Position","Normal","Vector","IndexOB","IndexMA",\
    "DiffCol","Denoising Normal","Denoising Albedo","Denoising Depth","UV"]],-1)
    return imgs



class TrainingDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
    super(TrainingDataset, self).__init__(cfg, name)
    self.max_padding = 16
    self.path = "/home/ascardigli/blender-3.2.2-linux-x64/suntemple/"
    self.num_images=1000
  def __len__(self):
    return self.num_images

  def __getitem__(self, index):
    # Get the input and target images
    input_name = "-"+str(index).zfill(4)+".png"
    target_name = "gd"+str(index).zfill(4)+".png"
    idxs = np.random.normal(loc=4*np.ones((720,1280,3)))
    idxs = np.round(idxs).astype(int)
    idxs[idxs<0]=0
    idxs[idxs>15] = 15
    sampling = np.round(np.random.rand(8,720,1280,3)*idxs).astype(int)
    samples = np.concatenate([get_ith_image(self.path,i,index) for i in range(16)],0)
    samples = np.take_along_axis(samples,sampling,0)
    input_image = np.transpose(samples,(1,2,3,0))
    height = input_image.shape[0]
    width  = input_image.shape[1]

    # Generate a random crop
    sy = sx = self.tile_size
    if rand() < 0.1:
      # Randomly zero pad later to avoid artifacts for images that require padding
      sy -= randint(self.max_padding)
      sx -= randint(self.max_padding)
    oy = randint(height - sy + 1)
    ox = randint(width  - sx + 1)

    target_image = get_truth(self.path,index)
    aux = get_aux(self.path,index)
    input_image=np.concatenate([input_image,aux],-1)
    color_order = randperm(3)
    input_image=input_image[:,:,color_order,:]

    input_image  = input_image [oy:oy+sy, ox:ox+sx]
    target_image = target_image[oy:oy+sy, ox:ox+sx]

    # Randomly transform the tiles to improve training quality
    if rand() < 0.5:
      # Flip vertically
      input_image  = np.flip(input_image,  0)
      target_image = np.flip(target_image, 0)

    if rand() < 0.5:
      # Flip horizontally
      input_image  = np.flip(input_image,  1)
      target_image = np.flip(target_image, 1)

    if rand() < 0.5:
      # Transpose
      input_image  = np.swapaxes(input_image,  0, 1)
      target_image = np.swapaxes(target_image, 0, 1)
      sy, sx = sx, sy

    input_image=input_image.reshape(*input_image.shape[:2],-1)

    # Zero pad the tiles (always makes a copy)
    pad_size = ((0, self.tile_size - sy), (0, self.tile_size - sx), (0, 0))
    input_image  = np.pad(input_image,  pad_size, mode='constant')
    target_image = np.pad(target_image, pad_size, mode='constant')

    # Randomly zero the main feature channels if there are auxiliary features
    # This prevents "ghosting" artifacts when the main feature is entirely black
    if rand() < 0.01:
      input_image[:, :, :24] = 0
      target_image[:] = 0

    # DEBUG: Save the tile
    #save_image('tile_%d.png' % index, target_image)

    # Convert the tiles to tensors
    return image_to_tensor(input_image), image_to_tensor(target_image)

## -----------------------------------------------------------------------------
## Validation dataset
## -----------------------------------------------------------------------------

class ValidationDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
    super(ValidationDataset, self).__init__(cfg, name)

    self.path = "/home/ascardigli/blender-3.2.2-linux-x64/suntemple/"

    self.num_images=400
    self.tiles = []

    if self.num_images == 0:
      return

    """
    # Split the images into tiles
    for sample_index in range(self.num_images):
      # Get the input image
      input_name,  _ = self.samples[sample_index]
      input_image, _ = self.images[input_name]

      # Get the size of the image
      height = input_image.shape[0]
      width  = input_image.shape[1]
      if height < self.tile_size or width < self.tile_size:
        error('image is smaller than the tile size')

      # Compute the number of tiles
      num_tiles_y = height // self.tile_size
      num_tiles_x = width  // self.tile_size

      # Compute the start offset for centering
      start_y = (height % self.tile_size) // 2
      start_x = (width  % self.tile_size) // 2

      # Add the tiles
      for y in range(num_tiles_y):
        for x in range(num_tiles_x):
          oy = start_y + y * self.tile_size
          ox = start_x + x * self.tile_size

          if self.main_feature == 'sh1':
            for k in range(0, 9, 3):
              ch = input_channel_indices[k:k+3] + input_channel_indices[9:]
              self.tiles.append((sample_index, oy, ox, ch))
          else:
            self.tiles.append((sample_index, oy, ox, input_channel_indices))
    """

      
  def __len__(self):
    return self.num_images

  def __getitem__(self, index):
    # Get the tile
 #   sample_index, oy, ox, input_channel_indices = self.tiles[index]
    index=index+1000
    sy = sx = self.tile_size

    input_name = "-"+str(index).zfill(4)+".png"
    target_name = "gd"+str(index).zfill(4)+".png"
    idxs = np.random.normal(loc=4*np.ones((720,1280,3)))
    idxs = np.round(idxs).astype(int)
    idxs[idxs<0]=0
    idxs[idxs>15] = 15
    sampling = np.round(np.random.rand(8,720,1280,3)*idxs).astype(int)
    samples = np.concatenate([get_ith_image(self.path,i,index) for i in range(16)],0)
    samples = np.take_along_axis(samples,sampling,0)
    input_image = np.transpose(samples,(1,2,3,0))
    target_image = get_truth(self.path,index)
    aux = get_aux(self.path,index)
    input_image=np.concatenate([input_image,aux],-1)
    input_image=input_image.reshape(*input_image.shape[:2],-1)

#    input_image  = input_image [oy:oy+sy, ox:ox+sx]
#    target_image = target_image[oy:oy+sy, ox:ox+sx]

    return image_to_tensor(input_image.copy()), image_to_tensor(target_image.copy())
