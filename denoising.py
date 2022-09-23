import numpy as np
from pypfm import PFMLoader
from PIL import Image
import os

albedo="~DiffCol0004.png"
normal= "~Normal0004.png"

def tensor_to_pfm(img,name):
  img =np.array(img,dtype=np.float32)/256.
  shape = img.shape()[1:]
  loader = PFMLoader((shape[1], shape[0]), True, compress=False)
  loader.save_pfm(name+".pfm", img)

def path_to_pfm(path,name):
  img = Image.open(albedo)
  tensor_to_pfm(img,name)


def pfm_to_tensor(path):
  loader = PFMLoader(color= True, compress=False)
  image = loader.load_pfm(path+".pfm")
  return (image*255).astype(np.uint8)

def denoiser(color,alb,nrm):
  os.system(" ~/oidn-1.4.3.x86_64.linux/bin/./oidnDenoise --ldr "+color+".pfm --alb "+alb+".pfm --nrm "+nrm+".pfm -o out.pfm")


def initialise(path):
  path_to_pfm(path+albedo,"albedo")
  path_to_pfm(path+normal,"normal")

def denoise(img):
  tensor_to_pfm(img,"color")
  denoiser("color","albedo","normal")
  return pfm_to_tensor("out")

