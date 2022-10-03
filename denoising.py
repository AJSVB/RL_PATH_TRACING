import numpy as np
from pypfm import PFMLoader
from PIL import Image
import os

albedo="~DiffCol0004.png"
normal= "~Normal0004.png"
import time
HEIGHT=720 
WIDTH =   1280

def r(i):
  return i[-HEIGHT:,-WIDTH:,:]


def tensor_to_pfm(img,name,rescale=False):

  img =r(np.array(img,dtype=np.float32))
  if rescale:
    img = img/255.
  shape = img.shape[1:]
  loader = PFMLoader((shape[1], shape[0]), True, compress=False)
  loader.save_pfm("/home/ascardigli/RL_PATH_TRACING/tmp/"+name+".pfm", img)

def path_to_pfm(path,name):
  img = Image.open(path)
  tensor_to_pfm(img,name,True)


def pfm_to_tensor(path):
  loader = PFMLoader(color= True, compress=False)
  image = loader.load_pfm(path+".pfm")
  return image

def denoiser(color,alb,nrm,pid):
  a="/home/ascardigli/RL_PATH_TRACING/"
#  b=time.time()
  os.system("~/oidn-1.4.3.x86_64.linux/bin/./oidnDenoise --ldr "+a+"tmp/"+color+".pfm --alb "+a+"tmp/"+alb+".pfm --nrm "+a+"tmp/"+nrm+".pfm -v 0 -o "+a+"tmp/"+pid+".pfm  > /dev/null")
#  print("denoising real time" + str(time.time()-b))

def initialise(path):
  path_to_pfm(path+albedo,"albedo")
  path_to_pfm(path+normal,"normal")

def denoise(img,pid):
  tensor_to_pfm(img,"color")
  denoiser("color","albedo","normal",pid)
  return pfm_to_tensor("/home/ascardigli/RL_PATH_TRACING/tmp/"+pid)

