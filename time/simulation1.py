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

L=1


class PhysicSimulation:

    def __init__(self,spp, sppps=.1,HEIGHT=480,WIDTH=730,sel=None):

        self.model,self.data,self.criterion,self.optimizer,self.scheduler = \
sel.model,sel.data,sel.criterion,sel.optimizer,sel.scheduler
        self.dataset = np.transpose(self.data.data(0),(1,2,3,0))[:720,:720]
        self.add, self.gd = self.data.get(0)
        self.add = self.add.permute(1,2,0).unsqueeze(-1).to(dtype=torch.float32)
        self.gd=self.gd.to(dtype=torch.float32)
        self.gd = self.gd.cuda(0)
        self.HEIGHT =  HEIGHT
        self.WIDTH =   WIDTH
        self.sppps = sppps
        self.shape=self.dataset.shape
        self.dataset=self.dataset.reshape( -1,*self.dataset.shape[2:])
        self.reset()
        self.a=torch.Tensor(range(720)).unsqueeze(0).repeat(720,1).unsqueeze(-1)/720.
        self.b=torch.Tensor(range(720)).unsqueeze(1).repeat(1,720).unsqueeze(-1)/720.
    def reset(self):
        self.observations = -1 * torch.ones([self.HEIGHT*self.WIDTH,3,8])
        self.s = torch.ones([self.HEIGHT, self.WIDTH,1], dtype = torch.float32)*1e-8 # 
        self.updated=False
        self.count=0
        self.loss=0

    def round_retain_sum(self,x):
     N = np.round(np.sum(x)).astype(int)
     y = x.astype(int)
     M=np.sum(y)
     K = N - M 
     z = x-y 
     if K!=0:
       idx = np.argsort(z)[-K:]
       y[idx] +=1     
     return y


    def simulate(self, x):
        x = x - np.min(x)
        x=x.flatten().astype(np.float64)
        x=x*self.sppps*self.WIDTH*self.HEIGHT/L/L/sum(x)
        s=np.array(self.round_retain_sum(x))
        self.count+=1
        if random.random()>.99:
            print(x)
            print(s)
            print(np.sum(s))
        s[s<0]=0
        s[s>7] = 7
        self.observations = self.data.generate(self.dataset,s)
        self.updated=False
        self.s = self.out(torch.Tensor(s/7.))

    def out(self,data):
        return data.view(self.HEIGHT,self.WIDTH,-1)#.type(torch.float16)    

    def render(self):        
      if not self.updated:
          self.optimizer.zero_grad()
          input = self.data.__getitem__(0,self.dataset.reshape(self.shape))
          self.denoised= self.model(input.cuda(0))
          loss = self.criterion(self.denoised, self.gd.unsqueeze(0))
          loss.backward()
          self.optimizer.step()
          self.scheduler.step()
          self.denoised = 1-torch.nn.ReLU()(1-torch.nn.ReLU()(self.denoised.detach())).cpu()
          if random.random()<1e-3:
           input=input[0].detach().cpu()
           for i in range(len(input)):
            plt.imshow(input[i])
            plt.savefig("images/"+str(i)+".png")
            plt.clf()
           plt.imshow(self.gd.permute(1,2,0).detach().cpu())
           plt.savefig("images/target.png")
           plt.clf()
           plt.imshow(self.denoised[0].permute(1,2,0).detach().cpu())
           plt.savefig("images/out.png")
          self.loss= loss.detach().cpu()
          self.denoised = self.denoised.reshape(3,-1).permute(1,0)

      self.updated=True
      return self.denoised


    def observe(self):
        a=self.render()
#        rendersquared = self.observations**2
        temp = torch.cat((
self.out(self.observations),\
#self.out(self.indexes/torch.max(self.indexes)), \
#self.out((self.variance - rendersquared)),\
self.out(self.add), \
self.out(a), \
  self.a,self.b,self.s \
 ),axis=-1).permute(2,0,1)
        return temp, self.gd


class Spec:
   def __init__(self,max_episode_steps):
    self.max_episode_steps = max_episode_steps
    self.id = "3Drenderingenv"

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import ray

def MultiSSIM(a,b,gpu_id):
  if gpu_id==-1:
    a[0] = a[0].type(torch.float32)
    b[0] = b[0].type(torch.float32)
    d = torch.cat([c.permute([2,0,1]).unsqueeze(0) for c in a],0)
    e = torch.cat([c for c in b],0)
    loss=MS_SSIM(data_range=1,size_average=False)
    return loss(d,e)

  else:
    d = torch.cat([c.permute([2,0,1]).unsqueeze(0) for c in a],0).cuda(gpu_id)
    e = torch.cat([c for c in b],0).cuda(gpu_id)
    loss=MS_SSIM(data_range=1,size_average=False).cuda(gpu_id)
    return loss(d,e.unsqueeze(0)).cpu() 




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
    self.max = int((self.spp)/self.sppps)  
    self.id = env_config.vector_index
    self.model,self.data,self.criterion,self.optimizer,self.scheduler = train.main_worker()
    self.model=self.model.to("cuda:0")
    self.criterion = self.criterion.to("cuda:0")
    self.simulation = PhysicSimulation(self.spp,self.sppps,self.HEIGHT,self.WIDTH,self)
    self.action_space = spaces.Box(low=0,high=1,shape=(int(self.HEIGHT*self.WIDTH/L/L),))
    self.observation_space = spaces.Box(low=-1, high=1, shape=
                    (int(self.HEIGHT/L),int(self.WIDTH/L),56), dtype=np.float32) #MACHINE PRECISION
    self.spec = Spec(self.max)
    self.top=0




  def step(self, action):
    old = self.simulation.out(self.simulation.render())
    i=-0 #works with 0 outside of tune.py TODO
    self.simulation.simulate(action)
    observation,gd = self.simulation.observe()
    new = self.simulation.out(self.simulation.render())
    base = self.simulation.dataset[:,:,:4].mean(-1)
    base = torch.Tensor(base).reshape(self.HEIGHT,self.WIDTH,3)
    baseline = MultiSSIM([base],[gd],i)[0]
    old = MultiSSIM([old], [gd],i)[0]
    new = MultiSSIM([new], [gd],i)[0]
    if  self.top<new and random.random()<.01:
        print("baseline " + str(baseline.item()))
        print("new "+str(new.item()))
        print("denoiser "+str(self.simulation.loss.item()))
        print()
        self.top = new
        if self.top>.9:
         self.insight()
    reward = 10**(new)
    done = self.spec.max_episode_steps <= self.simulation.count
    return observation.numpy().transpose(1,2,0),reward.detach().numpy(),done,{}

  def insight(self): 
    img= self.simulation.s
    te=str(self.top.item())
    save(img,"/home/ascardigli/RL_PATH_TRACING/tmp/"+te+".png")
  def reset(self):
    self.simulation = PhysicSimulation(self.spp,self.sppps,self.HEIGHT,self.WIDTH,self)
    temp ,_= self.simulation.observe()
    return temp.numpy().transpose(1,2,0)
    
def save(data,name):
    img= T.ToPILImage()(data.permute([2,0,1]).type(torch.float32))
    img.save(name)
