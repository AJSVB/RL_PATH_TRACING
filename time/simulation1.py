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


def p(x,y):
   a=1
#   print(x+str(y))


class PhysicSimulation:

    def __init__(self,spp, sppps=.1,HEIGHT=480,WIDTH=730,sel=None):

        self.model,self.data,self.criterion,self.optimizer,self.scheduler = \
sel.model,sel.data,sel.criterion,sel.optimizer,sel.scheduler
        self.HEIGHT =  HEIGHT
        self.WIDTH =   WIDTH
        self.sppps = sppps
        self.reset()
        self.shape=self.dataset.shape

    def reset(self):
        self.observations = -1 * torch.ones([8,3,self.HEIGHT,self.WIDTH])
        self.updated=True
        self.denoised = torch.zeros([1,3,self.HEIGHT,self.WIDTH]).cuda(0)
        self.count=0
        self.loss=0
        self.new(0)
        self.s = None
        self.state = -1 * torch.ones([64,self.HEIGHT,self.WIDTH]).cuda(0)



    def new(self,i):
        self.dataset = self.data.data(i)
        self.add, self.gd = self.data.get(i)
        self.add = torch.Tensor(self.add).permute(2,0,1).cuda(0) #necessary
        self.gd = torch.Tensor(self.gd).permute(2,0,1).cuda(0) #necessary

    def round_retain_sum(self,x,N):
     N = np.round(N).astype(int)
     y = x.type(torch.int)
     M=torch.sum(y)
     K = N - M 
     z = x-y 
     if K!=0:
       idx = torch.topk(z,K,sorted=False).indices
       y[idx] +=1     
     return y

    def simulate(self, x):
        b=time.time()
        x=torch.Tensor(x).cuda(0)
        x = x - torch.min(x) 
        x=torch.flatten(x).type(torch.float64)
        N= torch.sum(x)
        temp = self.sppps*self.WIDTH*self.HEIGHT 
        x=x*temp/N
        N=temp
        s= self.round_retain_sum(x,N)
        if random.random()>.999:
            print(x.cpu())
            print(s.cpu())
            print(torch.sum(s).cpu())
        s[s<0]=-1
        s[s>7] = 7
        self.s=s
        a=time.time()
        self.observations = self.data.generate(self.dataset,s,self.count)
        self.count+=1
        self.updated=False

    def out(self,data):
        return data.reshape(-1, self.HEIGHT,self.WIDTH)

    def render(self):        
      if not self.updated:
          a=time.time()
          self.optimizer.zero_grad()
          if self.count !=1:
              self.state = self.data.translation(self.count-1,self.state)
          m1=self.observations.reshape(-1,*self.shape[-2:])
          m2=self.add
          m3=self.state
          input= torch.cat((m1,m2,m3),0).unsqueeze(0)
          self.denoised, self.state= self.model(input)
          loss = self.criterion(self.denoised, self.gd.unsqueeze(0)) #*2
          if self.count<80:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
          self.denoised = torch.clip(self.denoised.detach(),0,1)
          self.state = self.state.detach()

          if random.random()<1e-4:
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
           plt.clf()
           plt.imshow(self.s.detach().cpu().reshape(720,720))
           plt.savefig("images/hitmap.png")



      self.updated=True
      return self.denoised


    def observe(self):
#        temp = np.concatenate((
#self.out(self.observations),\
#self.out(self.add), \
# ),axis=0)
        return self.state.cpu(), self.gd


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
    d = torch.cat([c.unsqueeze(0) for c in a],0)
    e = torch.cat([c for c in b],0)
    loss=MS_SSIM(data_range=1,size_average=False)
    return loss(d,e)

  else:
    d = torch.cat([c.unsqueeze(0) for c in a],0)
    e = torch.cat([c for c in b],0)
    loss=MS_SSIM(data_range=1,size_average=False)
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
    self.max = int((self.spp)/self.sppps)*100
    self.id = env_config.vector_index
    self.model,self.data,self.criterion,self.optimizer,self.scheduler = train.main_worker()
    self.model=self.model.to("cuda:0")
    self.criterion = self.criterion.to("cuda:0")
    self.simulation = PhysicSimulation(self.spp,self.sppps,self.HEIGHT,self.WIDTH,self)
    self.action_space = spaces.Box(low=0,high=1,shape=(int(self.HEIGHT*self.WIDTH),))
    self.observation_space = spaces.Box(low=-1.0001, high=1, shape=
                    (64,int(self.HEIGHT),int(self.WIDTH)), dtype=np.float32) #MACHINE PRECISION
    self.spec = Spec(self.max)
    self.top=0
    self.a=time.time()
  def step(self, action):
    t = str(time.time()-self.a)
    self.a = time.time()
    self.simulation.new(self.simulation.count)
    old = self.simulation.out(self.simulation.render())
    i=-0 #works with 0 outside of tune.py TODO
    self.simulation.simulate(action)
    observation,gd = self.simulation.observe()
    new = self.simulation.out(self.simulation.render())
    base = self.simulation.dataset[:4].mean(0)
    baseline = MultiSSIM([base],[gd],i)[0]

    if self.bool:
     te = str(self.simulation.count)
     save(base.cpu(),"images/base"+te+".png")
     save(new.cpu(),"images/new"+te+".png")
     save(gd.cpu(),"images/gd"+te+".png")



    old = MultiSSIM([old], [gd],i)[0]
    new = MultiSSIM([new], [gd],i)[0]
    if  False and self.top<new:
        print("baseline " + str(baseline.item()))
        print("new "+str(new.item()))
        print("denoiser "+str(self.simulation.loss.detach().cpu().item()))
        print("time "+ str(time.time()-a))
        print(self.simulation.count)
        print()
        self.top = new
        if self.top>.95:
         save(base,"images/baseline.png")
         self.insight()



    reward = 10**(new)
    done = self.spec.max_episode_steps <= self.simulation.count
    print("outside step " + t + ", inside step "+str(time.time()-self.a))
    self.a=time.time()
    return observation.numpy(),reward.detach().numpy(),done,{}
 
  def f(self,a):
    a=a.numpy()
    print(a.shape)
    print(np.min(a))
    print(np.max(a))
    print(a.dtype)



  def insight(self): 
    img= self.simulation.s.reshape(self.HEIGHT,self.WIDTH,1)*1./np.max(self.simulation.s)
    te=str(self.top.item())
    save(img.astype(np.float32),"images/"+te+".png")
  def reset(self):
    a=time.time()
    self.bool=False
    if random.random()<0.01:
      self.bool=True
    self.simulation = PhysicSimulation(self.spp,self.sppps,self.HEIGHT,self.WIDTH,self)
    temp ,_= self.simulation.observe()
#    print(self.f(temp))
    return temp.numpy()
    
def save(data,name):
    img= T.ToPILImage()(data)
    if img.mode != 'RGB':
     img = img.convert('RGB')
    img.save(name)
