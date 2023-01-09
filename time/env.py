from physicalsimulation import PhysicSimulation
import math
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import time
import torchvision.transforms as T
import random
import torch
import unet1
from filelock import FileLock 
import os
from training import  train
import matplotlib.pyplot as plt
import random
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import mean_squared_error
L=1
psnr = PeakSignalNoiseRatio().cuda(0)

def p(x,y):
   a=1

list = []

import random
from torchvision.transforms import *

from torchvision.transforms import functional as F

class Spec:
   def __init__(self,max_episode_steps):
    self.max_episode_steps = max_episode_steps
    self.id = "3Drenderingenv"

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import ray

import gym
from gym import Env, spaces
import numpy as np

class CustomEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, env_config):
    super(CustomEnv, self).__init__()
    self.spp = env_config['spp']
    self.sppps = env_config["sppps"]
    self.HEIGHT = 720 
    self.WIDTH =   720

    self.max = 20
    out_space=41
    self.mode = env_config["mode"]
    if self.mode == "ntas":
     self.model,self.data,self.criterion,self.optimizer,self.scheduler = train.main_worker(12,3)
     out_space=12
    elif "notp" in self.mode:
     self.model,self.data,self.criterion,self.optimizer,self.scheduler = train.main_worker(33,0)
     out_space=9
    elif self.mode == "dasr":
     self.model,self.data,self.criterion,self.optimizer,self.scheduler = train.main_worker(12,0)
     out_space=9
    else:

     self.model,self.data,self.criterion,self.optimizer,self.scheduler = train.main_worker()

    self.model=self.model.to("cuda:0")
    self.criterion = self.criterion.to("cuda:0")


    self.offset=0
    self.simulation = PhysicSimulation(self.spp,self.sppps,self.HEIGHT,self.WIDTH,self,self.offset,self.mode)
    self.action_space = spaces.Box(low=0,high=1,shape=(int(self.HEIGHT*self.WIDTH),))
    self.observation_space = spaces.Box(low=-1.0001, high=1.0001, shape=
                    (out_space,int(self.HEIGHT),int(self.WIDTH)), dtype=np.float32) #MACHINE PRECISION
    self.spec = Spec(self.max)
    self.top=0
    self.a=time.time()
    self.mses = []
    self.psnrs = []

    with open("comp/"+str(self.spp)+'mses'+self.mode+'.txt', 'w') as fp:
        fp.write("\n")
    with open("comp/"+str(self.spp)+'psnrs'+self.mode+'.txt', 'w') as fp:
        fp.write("\n")

    self.time=time.time()
    self.sum=0
    self.cntr=0


  def step(self, action):
    self.sum+=time.time() - self.time
    self.time= time.time()
    self.cntr+=1
    if self.cntr ==1000:
      print(self.sum/1000)
      self.cntr=0
      self.sum=0   

    self.simulation.new(self.simulation.count)
    self.simulation.simulate(action)
    new = self.simulation.out(self.simulation.render())
    observation,gd = self.simulation.observe()
    loss= self.simulation.loss
    new1=1-loss

    if  self.bool:
     te = str((self.simulation.offset%100)+self.simulation.count)
     save(new.cpu(),"images/"+str(self.spp)+"new"+te+"_"+str(self.mode)+".png")
  
     self.mses.append(mean_squared_error(new,gd).cpu())
     self.psnrs.append(psnr(new,gd).cpu())

    reward = 10**(new1)
    done = self.spec.max_episode_steps <= self.simulation.count
    return observation.numpy(),reward.detach().numpy(),done,{}
 
  def insight(self): 
    img= self.simulation.s.reshape(self.HEIGHT,self.WIDTH,1)*1./np.max(self.simulation.s)
    te=str(self.top.item())
    save(img.astype(np.float32),"images/"+te+".png")
  def reset(self):
    self.bool=False
    if self.offset%self.simulation.number >= 800 and self.offset%self.simulation.number<900 : #was between 800 and 900
      self.bool=True
    self.simulation = PhysicSimulation(self.spp,self.sppps,self.HEIGHT,self.WIDTH,self,int((self.offset//20)*20) %self.simulation.number,self.mode)
    temp ,_= self.simulation.observe()

    if self.bool:
     with open("comp/"+str(self.spp)+'mses'+self.mode+'.txt', 'a') as fp:
         fp.write("\n".join(str(item.item()) for item in self.mses))
         fp.write("\n")
     with open("comp/"+str(self.spp)+'psnrs'+self.mode+'.txt', 'a') as fp:
         fp.write("\n".join(str(item.item()) for item in self.psnrs))
         fp.write("\n")
    self.mses=[]
    self.psnrs=[]
    self.offset+=5
    return temp.numpy()
    
def save(data,name):
    img= T.ToPILImage()(data)
    if img.mode != 'RGB':
     img = img.convert('RGB')
    img.save(name)
	
