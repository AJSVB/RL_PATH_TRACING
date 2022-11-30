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
#   print(x+str(y))

#L1 = torch.nn.L1Loss()

class PhysicSimulation:

    def __init__(self,spp, sppps=.1,HEIGHT=480,WIDTH=730,sel=None):

        self.model,self.data,self.criterion,self.optimizer,self.scheduler = \
sel.model,sel.data,sel.criterion,sel.optimizer,sel.scheduler
        self.HEIGHT =  HEIGHT
        self.WIDTH =   WIDTH
        self.sppps = sppps
        self.offset=0
        self.reset()
        self.shape=self.dataset.shape
        self.i = sel.i
        self.number = 1200

    def reset(self):
        self.observations = -1 * torch.ones([8,3,self.HEIGHT,self.WIDTH])
        self.updated=True
        self.denoised = torch.zeros([1,3,self.HEIGHT,self.WIDTH]).cuda(0)
        self.count=0
        self.loss=0
        self.new(0)
        self.s = None
        self.state = -1 * torch.ones([32,self.HEIGHT,self.WIDTH]).cuda(0)

    def new(self,i):
        self.dataset = self.data.data(i+self.offset)
        self.add, self.gd = self.data.get(i+self.offset)
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
        s[s>8] = 8
        self.s=s
        a=time.time()
        self.observations = self.data.generate(self.dataset,s,self.count) # TODO - this I think :)
        self.count+=1
        self.updated=False

    def out(self,data):
        return data.reshape(-1, self.HEIGHT,self.WIDTH)

    def render(self):        
      if not self.updated:
          a=time.time()
          self.optimizer.zero_grad()
          m1=self.observations.reshape(-1,*self.shape[-2:])
          m2=self.add
          m3=self.state
          input= torch.cat((m1,m2,m3),0).unsqueeze(0)
          self.denoised, self.state= self.model(input)
          loss = self.criterion(self.denoised, self.gd.unsqueeze(0)) * self.i 

          if True: #self.offset<800 or self.offset>=900:
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
        if self.count >1:
            self.state = self.data.translation(self.count-1,self.state)
        m2=self.add
        m3=self.state.detach()
        input= torch.cat((m2,m3),0)
        return input.cpu(), self.gd


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

    self.max = 100 


    self.id = env_config.vector_index
    self.i = env_config["i"]
    self.model,self.data,self.criterion,self.optimizer,self.scheduler = train.main_worker()
    self.model=self.model.to("cuda:0")
    self.criterion = self.criterion.to("cuda:0")
    self.offset=0
    self.simulation = PhysicSimulation(self.spp,self.sppps,self.HEIGHT,self.WIDTH,self)
    self.action_space = spaces.Box(low=0,high=1,shape=(int(self.HEIGHT*self.WIDTH),))
    self.observation_space = spaces.Box(low=-1.0001, high=1, shape=
                    (41,int(self.HEIGHT),int(self.WIDTH)), dtype=np.float32) #MACHINE PRECISION
    self.spec = Spec(self.max)
    self.top=0
    self.a=time.time()
    self.mses = []
    self.psnrs = []

    with open(str(self.spp)+'mses.txt', 'w') as fp:
        fp.write("\n")
    with open(str(self.spp)+'psnrs.txt', 'w') as fp:
        fp.write("\n")



  def step(self, action):
    self.simulation.new(self.simulation.count)
    old = self.simulation.out(self.simulation.render())
    i=-0 #works with 0 outside of tune.py TODO
    self.simulation.simulate(action)
    observation,gd = self.simulation.observe()
    new = self.simulation.out(self.simulation.render())
    base = self.simulation.dataset[:4].mean(0)
    baseline = MultiSSIM([base],[gd],i)[0]
    old1 = MultiSSIM([old], [gd],i)[0]
    new1 = MultiSSIM([new], [gd],i)[0]

#    baseline = -L1(base,gd).cpu()
#    old1 = -L1(old, gd).cpu()
#    new1 = -L1(new, gd).cpu()



    if self.bool and self.simulation.count==1:
     self.bool = new1>self.top
     if self.bool:
       self.top = new1

    if self.bool:
     te = str(self.simulation.count)
     save(base.cpu(),"images/"+str(self.spp)+"base"+te+".png")
     save(new.cpu(),"images/"+str(self.spp)+"new"+te+".png")
     save(gd.cpu(),"images/"+str(self.spp)+"gd"+te+".png")
  
     self.mses.append(mean_squared_error(new,gd).cpu())
     self.psnrs.append(psnr(new,gd).cpu())

    reward = 10**(new1)
    done = self.spec.max_episode_steps <= self.simulation.count
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
    print("res")
    self.bool=False
    if self.offset%self.simulation.number >= 800 and self.offset%self.simulation.number<900 :
      self.bool=True
    self.simulation = PhysicSimulation(self.spp,self.sppps,self.HEIGHT,self.WIDTH,self)
    self.offset+=5 
    self.simulation.offset = int((self.offset//100)*100) %self.simulation.number
    temp ,_= self.simulation.observe()


    if self.bool:
     with open(str(self.spp)+'mses.txt', 'a') as fp:
         fp.write("\n".join(str(item.item()) for item in self.mses))
         fp.write("\n")
     with open(str(self.spp)+'psnrs.txt', 'a') as fp:
         fp.write("\n".join(str(item.item()) for item in self.psnrs))
         fp.write("\n")

    self.mses=[]
    self.psnrs=[]

    return temp.numpy()
    
def save(data,name):
    img= T.ToPILImage()(data)
    if img.mode != 'RGB':
     img = img.convert('RGB')
    img.save(name)
