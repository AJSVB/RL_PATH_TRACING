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
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x

def get_truth(path,frame_number):
    image= Image.open(path + "gd"+str(frame_number).zfill(4)+".png")
    x = TF.to_tensor(image)
    return x


def get_aux(path,frame_number):
    import minexr
    f=path + "add"+str(frame_number).zfill(4)+".exr"
    with open(f, 'rb') as fp:
        reader = minexr.load(fp)
        print(reader.shape)
    x = TF.to_tensor(image)
    return x



class TrainingDataset(PreprocessedDataset):
  def __init__(self, cfg, name):
    super(TrainingDataset, self).__init__(cfg, name)
    self.max_padding = 16
    self.path = "/home/ascardigli/blender-3.2.2-linux-x64/suntemple/"
    self.num_images=1000
  def __len__(self):
    return 1000

  def __getitem__(self, index):
    # Get the input and target images
    input_name = "-"+str(index).zfill(4)+".png"
    target_name = "gd"+str(index).zfill(4)+".png"
    idxs = torch.normal(mean=4*torch.ones(3,720,1280))
    idxs = torch.round(idxs).to(int)
    idxs[idxs<0]=0
    idxs[idxs>15] = 15
    sampling = torch.round(torch.rand((8,3,720,1280))*idxs).to(int)
    samples = torch.cat([get_ith_image(self.path,i,index) for i in range(16)],0)
    samples = torch.take_along_dim(samples,sampling,0)
    print(samples.shape)
    input_image = samples.permute(2,3,1,0)
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


    gd = get_truth(self.path,index).permute(1,2,0)

    aux = get_aux(self.path,index)




    color_order = randperm(3)
    input_image=input_image[:,:,color_order,:]




    # Crop the input and target images
    if self.clean_aux:
      # Get the auxiliary features from the target image
      aux_channel_indices = input_channel_indices[self.num_main_channels:]
      input_image  = input_image [oy:oy+sy, ox:ox+sx, target_channel_indices]
      aux_image    = target_image[oy:oy+sy, ox:ox+sx, aux_channel_indices]
      target_image = target_image[oy:oy+sy, ox:ox+sx, target_channel_indices]
      input_image  = np.concatenate((input_image, aux_image), axis=2)
    else:
      # Get the auxiliary features from the input image
      input_image  = input_image [oy:oy+sy, ox:ox+sx, input_channel_indices]
      target_image = target_image[oy:oy+sy, ox:ox+sx, target_channel_indices]

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

    # Zero pad the tiles (always makes a copy)
    pad_size = ((0, self.tile_size - sy), (0, self.tile_size - sx), (0, 0))
    input_image  = np.pad(input_image,  pad_size, mode='constant')
    target_image = np.pad(target_image, pad_size, mode='constant')

    # Randomly zero the main feature channels if there are auxiliary features
    # This prevents "ghosting" artifacts when the main feature is entirely black
    if self.aux_features and rand() < 0.01:
      input_image[:, :, 0:self.num_main_channels] = 0
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

    self.path = "~/blender-3.2.2-linux-x64/suntemple/"
    self.num_images=400

      
  def __len__(self):
    return 400

  def __getitem__(self, index):
    # Get the tile
    sample_index, oy, ox, input_channel_indices = self.tiles[index]
    sy = sx = self.tile_size

    # Get the input and target images
    input_name, target_name = self.samples[sample_index]
    input_image,  _ = self.images[input_name]
    target_image, _ = self.images[target_name]

    # Get the indices of target channels
    target_channel_indices = input_channel_indices[:self.num_main_channels]

    # Crop the input and target images
    if self.clean_aux:
      # Get the auxiliary features from the target image
      aux_channel_indices = input_channel_indices[self.num_main_channels:]
      input_image  = input_image [oy:oy+sy, ox:ox+sx, target_channel_indices]
      aux_image    = target_image[oy:oy+sy, ox:ox+sx, aux_channel_indices]
      target_image = target_image[oy:oy+sy, ox:ox+sx, target_channel_indices]
      input_image  = np.concatenate((input_image, aux_image), axis=2)
    else:
      # Get the auxiliary features from the input image
      input_image  = input_image [oy:oy+sy, ox:ox+sx, input_channel_indices]
      target_image = target_image[oy:oy+sy, ox:ox+sx, target_channel_indices]

    # Convert the tiles to tensors
    # Copying is required because PyTorch does not support non-writeable tensors
    return image_to_tensor(input_image.copy()), image_to_tensor(target_image.copy())
