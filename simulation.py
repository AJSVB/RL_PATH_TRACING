
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
import unet
import denoising
from filelock import FileLock 
import os

def norm(a):
    return (a-np.min(a))/(np.max(a)-np.min(a))

def ground_truth(path,number_images=1000,frame_number=1,HEIGHT=720,WIDTH=1280,name=""):
    dataset = torch.cat([get_ith_image(path,i,frame_number,HEIGHT,WIDTH) for i in range(number_images)],0)
    dataset = dataset.mean(0)
    img= T.ToPILImage()(dataset)
    img.save(path+name)

def get_truth(path,HEIGHT=480,WIDTH=640):
    image= Image.open(path)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x[:,:,-HEIGHT:,-WIDTH:]


def get_ith_image(path,i,frame_number = 1,HEIGHT=480,WIDTH=640):
    image = Image.open(path+str(frame_number).zfill(4) + "-" + str(i).zfill(5)+'.png0001.png')
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x[:,:,-HEIGHT:,-WIDTH:]

def aggregate_by_pixel(path,number_images,frame_number = 1,HEIGHT=480,WIDTH=640):
    dataset = torch.cat([get_ith_image(path,i,frame_number,HEIGHT,WIDTH) for i in range(number_images)],0)
    dataset = dataset.permute([2,3,1,0])
    return dataset

def cached(list):
    return torch.cat(list,0).permute([2,3,1,0])


def get_add(path,detail):
    image = Image.open(path+"~"+detail+'0004.png')
    x = TF.to_tensor(image)
    if x[0,:].equal(x[1,:]):
        x = x[0:1,:]
    return x.mean(0).unsqueeze(0)

def load_additional(path,frame_number=1,HEIGHT=480,WIDTH=640):
    dataset = torch.cat([get_add(path,a) for a in  ["Normal","DiffCol"]])
    dataset = dataset.permute([1,2,0])
    return dataset[-HEIGHT:,-WIDTH:]

class PhysicSimulation:
    def __init__(self,path,spp,frame_number=1, sppps=.1,list=None,add = None,HEIGHT=480,WIDTH=730,max=10,denoinsing=True):
        self.HEIGHT =  HEIGHT
        self.WIDTH =   WIDTH
        self.dataset = cached(list)
#        self.dataset=aggregate_by_pixel(path,self.max,frame_number,self.HEIGHT,self.WIDTH)
        self.sppps = sppps
        self.dataset=self.dataset.view( -1, *self.dataset.shape[2:])
        self.add = add
        self.max = max
        self.reset()
        self.updated=False
        self.denoising=denoising
    def reset(self):
        self.permutation = torch.randperm(self.max)
        self.observations = self.dataset[:,:,self.permutation[0]] #torch.zeros(self.dataset.shape[:2])
        self.indexes = torch.ones([self.HEIGHT, self.WIDTH], dtype = torch.int)
        self.indexes=self.indexes.view( -1, *self.indexes.shape[2:])
        self.variance = self.observations**2
        self.count = 1 #0
        self.updated=False
    def simulate(self, x):
        x=x.flatten()
        max = np.quantile(x,1-self.sppps)
        idx = np.where(x>=max)[0]
        indexes = self.indexes[idx] 
        self.indexes[idx]= indexes+1
        temp = self.dataset[idx,:,self.permutation[self.count]]
        indexes = indexes.unsqueeze(1).repeat(1,3)
        self.observations[idx,:] = (self.observations[idx,:]*indexes +  temp )/(indexes+1)
        self.variance[idx,:] = (self.variance[idx,:]*indexes + temp**2 )/(indexes+1)
        self.count+=1
        self.updated=False

    def out(self,data):
        return data.view(self.HEIGHT,self.WIDTH,*data.shape[1:]).numpy()    

    def render(self):        
      if self.denoising:
        if not self.updated:
          with FileLock('tmp/0.pfm.lock'):
            self.denoised= denoising.denoise(self.out(self.observations),str(0))
          self.updated=True
        return self.denoised
      else:
        return self.out(self.observations)


    def observe(self):
        rendersquared = self.observations**2
        temp = np.concatenate((self.out(self.observations.mean(-1).unsqueeze(-1)),self.out((self.indexes/self.max).unsqueeze(-1)), self.out((self.variance - rendersquared).mean(-1).unsqueeze(-1))),axis=-1)           
        return np.concatenate((temp,self.add, np.expand_dims(norm((self.out(self.observations)-self.render()).mean(-1)),-1)   ),axis=-1,dtype=np.float16)


class Spec:
   def __init__(self,max_episode_steps):
    self.max_episode_steps = max_episode_steps
    self.id = "3Drenderingenv"

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
def MultiSSIM(a,b):
    return ms_ssim(torch.Tensor(a).permute([2,0,1]).unsqueeze(0),torch.Tensor(b),data_range=1)



import gym
from gym import Env, spaces
import numpy as np

class CustomEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, env_config):
    super(CustomEnv, self).__init__()
    self.path = env_config["path"]
    self.frame_number = env_config["frame_number"]
    self.spp = env_config['spp']
    self.sppps = env_config["sppps"]
    self.denoising = env_config['denoising']
    self.HEIGHT = 720 
    self.WIDTH =   1280
    self.max = int(self.spp/self.sppps) - int(1/self.sppps)+1 #
    self.list = [get_ith_image(self.path,i,self.frame_number,self.HEIGHT,self.WIDTH) for i in range(self.max)]
    self.add = load_additional(self.path,1,self.HEIGHT,self.WIDTH)
    self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps,self.list,self.add,self.HEIGHT,self.WIDTH,self.max,self.denoising)
    self.action_space = spaces.Box(low=0,high=1,shape=(self.HEIGHT*self.WIDTH,))
    self.observation_space = spaces.Box(low=-1e-6, high=1, shape=
                    (self.HEIGHT,self.WIDTH,6), dtype=np.float16) #MACHINE PRECISION
    denoising.initialise("../datasets/temple/")

    self.spec = Spec(self.max)
    self.ground_truth = get_truth("../datasets/temple/"+"truth.png",self.HEIGHT,self.WIDTH)
    self.top = 0

  def step(self, action):
    old = MultiSSIM(self.simulation.render(), self.ground_truth)
    self.simulation.simulate(action)
    observation = self.simulation.observe()
    new = MultiSSIM(self.simulation.render(), self.ground_truth)
    if self.top<new:
        print(old)
        self.top = new
    reward = - old + new
    done = self.spec.max_episode_steps <= self.simulation.count
    
    return observation,reward.detach().numpy(),done, {"msssim":new.detach()}

    
  def reset(self):
    if random.random()>.99:
        img= self.simulation.indexes.unsqueeze(-1)
        norm = (img-torch.min(img))/(torch.max(img) - torch.min(img))
        save(self.simulation.out(norm),"tmp/"+str(random.random())+".png")
    self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps,self.list,self.add,self.HEIGHT,self.WIDTH,self.max,self.denoising)
    return self.simulation.observe()
    
  def render(self, mode='human', close=False):
    return self.simulation.render()

def save(data,name):
    img= T.ToPILImage()(torch.Tensor(data).permute([2,0,1]))
    img.save(name)
