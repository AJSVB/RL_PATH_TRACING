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

list = []
from torch.autograd import Function
from torch.autograd import Variable
import random
from torchvision.transforms import *

from torchvision.transforms import functional as F

size=(720,720)
scale=(0.08, 1.0)
ratio=(3.0 / 4.0, 4.0 / 3.0)
interpolation=InterpolationMode.BILINEAR


class Render(Function):
    @staticmethod
    def forward(ctx, x, sim):
#        x = x - torch.min(x)
        x=torch.flatten(x)
        M = sim.sppps*sim.WIDTH*sim.HEIGHT
        e = torch.exp(x)
        s = torch.sum(e)
        s = torch.floor(M*e/s).clamp(min=1)
        s[s<0]=-1
        s[s>8] = 8
        sim.s=s
        if random.random()<.01:
         print(torch.mean(s))
        observations = sim.data.generate(sim.dataset,s.type(torch.long),sim.count) 

        obs = observations.reshape(8,-1)
        mask = obs!=-1
        obs = (obs*mask).sum(dim=0)/mask.sum(dim=0)
        obs = obs.reshape(3,720,720)

        ctx.save_for_backward((sim.gd - obs)/s.reshape(1,720,720).expand(3,-1,-1))
        return obs


    @staticmethod
    def backward(ctx, dL_dout):
        dS_dn = ctx.saved_tensors[0]

        dL_dout = dL_dout.unsqueeze(0)

        dL_din = torch.sum(torch.mul(dL_dout ,Variable(dS_dn)), dim=1, keepdim=True)

        return tuple([dL_din] + [None]*7)



def get_params():
        height, width = 720,720 
        area = height * width
        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

class PhysicSimulation:

    def __init__(self,spp, sppps=.1,HEIGHT=480,WIDTH=730,sel=None,offset=0):
        self.model,self.data,self.criterion,self.optimizer,self.scheduler = \
sel.model,sel.data,sel.criterion,sel.optimizer,sel.scheduler
        self.HEIGHT =  HEIGHT
        self.WIDTH =   WIDTH
        self.sppps = sppps
        self.number = 1200
        self.offset = offset
        self.reset()
        self.new(-1)
        self.shape= [8,720,720]
        self.i = sel.i
        self.loss=0
        self.x=1
        self.y=1

    def reset(self):
        self.observations = -1 * torch.ones([8,3,self.HEIGHT,self.WIDTH])
        self.updated=True
        self.denoised = torch.zeros([1,3,self.HEIGHT,self.WIDTH]).cuda(0)
        self.count=-1
        self.loss=0
        self.s = None
        self.state = -1 * torch.ones([3,self.HEIGHT,self.WIDTH]).cuda(0)
        lis = []
        self.perm = lambda x:x
        if random.random()>.5:
         lis.append(T.functional.hflip)
         self.x=-1
        if random.random()>.5:
         lis.append(T.functional.vflip)
         self.y=-1
        if random.random()>1.5:
         i, j, h, w = get_params()
         self.perm = lambda x: F.resized_crop(x, i, j, h, w,size, interpolation)
        if self.inval():
         lis=[]
         self.perm = lambda x:x
        self.transform = T.Compose(lis)

    def new(self,i):
        transform = lambda x: self.perm(self.transform(x))
        if i ==-1:
         self.nextadd, self.gd = self.data.get(i+1+self.offset)
         self.nextadd = transform(torch.Tensor(self.nextadd).permute(2,0,1).cuda(0))

        else:
         self.dataset = transform(self.data.data(i+self.offset))
         self.add, self.gd = self.data.get(i+self.offset)
         self.add = transform(torch.Tensor(self.add).permute(2,0,1).cuda(0)) #necessary
         self.gd = transform(torch.Tensor(self.gd).permute(2,0,1).cuda(0)) #necessary
         self.nextadd, _ = self.data.get(i+1+self.offset)
         self.nextadd = transform(torch.Tensor(self.nextadd).permute(2,0,1).cuda(0))

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
        self.observations = Render.apply(x,self) 
        self.updated=False

    def inval(self):
      return self.offset>=800 and self.offset<900

    def out(self,data):
        return data.reshape(-1, self.HEIGHT,self.WIDTH)

    def render(self):        
      if not self.updated:
          self.optimizer.zero_grad()
          m1=self.observations.reshape(-1,*self.shape[-2:])
          m2=self.add
          m3=self.state
          input= torch.cat((m1,m2,m3),0).unsqueeze(0)
#          with torch.cuda.amp.autocast():
          self.denoised, _= self.model(input)
          self.state = self.denoised
          loss = self.criterion(self.denoised, self.gd.unsqueeze(0)) 
          if random.random()<.01: 
            print(loss)
#          if self.oldgd is not None:
#            loss+=self.criterion(self.denoised-self.olddenoised,self.gd.unsqueeze(0)-self.oldgd.unsqueeze(0))
          if not self.inval(): #self.offset<800 or self.offset>=900:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
          self.denoised = torch.clip(self.denoised.detach(),0,1)
          self.state = torch.clip(self.state.detach(),-1,1)
          if random.random()< 0.0001:
           t = str(self.offset+self.count-1)
#           print(t)
#           plt.imshow(m1.cpu().mean(0))
#           plt.savefig("images/"+t+"obs.png")
           plt.clf()
           plt.imshow(m2.cpu().mean(0))
           plt.savefig("images/"+t+"add.png")
           plt.clf()
           plt.imshow(m3.cpu().mean(0))
           plt.savefig("images/"+t+"stt.png")
           plt.clf()
           plt.imshow(self.gd.permute(1,2,0).detach().cpu())
           plt.savefig("images/"+t+"target.png")
           plt.clf()
#           plt.imshow(self.denoised[0].to(torch.float).permute(1,2,0).detach().cpu())
#           plt.savefig("images/"+t+"out.png")
#           plt.clf()
#           plt.imshow(self.s.detach().cpu().reshape(720,720))
#           plt.savefig("images/"+t+"hitmap.png")

      self.updated=True
      self.loss=loss.detach().cpu()
      return self.denoised


    def observe(self):
        self.state = self.state.to(torch.float)
        if self.count>-1:
         self.state = self.transform(self.state)
         self.state = self.data.translation(self.count+self.offset,self.state,self.perm ) #,\
# script.f(self.count-1+self.offset,self.transform))
         self.state = self.transform(self.state)
        m2=self.nextadd
        m3=self.state.detach()
        input= torch.cat((m2,m3),0)
        self.count+=1
        return input.cpu(), self.gd


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


    self.i = env_config["i"]
    self.model,self.data,self.criterion,self.optimizer,self.scheduler = train.main_worker(12,3)
    self.model=self.model.to("cuda:0")
    self.criterion = self.criterion.to("cuda:0")
    self.offset=0
    self.simulation = PhysicSimulation(self.spp,self.sppps,self.HEIGHT,self.WIDTH,self,self.offset)
    self.action_space = spaces.Box(low=0,high=1,shape=(int(self.HEIGHT*self.WIDTH),))
    self.observation_space = spaces.Box(low=-1.0001, high=1.0001, shape=
                    (12,int(self.HEIGHT),int(self.WIDTH)), dtype=np.float32) #MACHINE PRECISION
    self.spec = Spec(self.max)
    self.top=0
    self.a=time.time()
    self.mses = []
    self.psnrs = []

    with open("comp/"+str(self.spp)+'msesntas.txt', 'w') as fp:
        fp.write("\n")
    with open("comp/"+str(self.spp)+'psnrsntas.txt', 'w') as fp:
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
     save(new.cpu(),"images/"+str(self.spp)+"new"+te+".png")
  
     self.mses.append(mean_squared_error(new,gd).cpu())
     self.psnrs.append(psnr(new,gd).cpu())

    reward = 10**(new1)
    done = self.spec.max_episode_steps <= self.simulation.count
    return observation,reward,done,{}
 
  def grad(self):
    return self.simulation.grad

  def insight(self): 
    img= self.simulation.s.reshape(self.HEIGHT,self.WIDTH,1)*1./np.max(self.simulation.s)
    te=str(self.top.item())
    save(img.astype(np.float32),"images/"+te+".png")
  def reset(self):
    self.bool=False
    if self.offset%self.simulation.number >= 800 and self.offset%self.simulation.number<900 : #was between 800 and 900
      self.bool=True
    self.simulation = PhysicSimulation(self.spp,self.sppps,self.HEIGHT,self.WIDTH,self,int((self.offset//20)*20) %self.simulation.number)
    temp ,_= self.simulation.observe()

    if self.bool:
     with open("comp/"+str(self.spp)+'msesntas.txt', 'a') as fp:
         fp.write("\n".join(str(item.item()) for item in self.mses))
         fp.write("\n")
     with open("comp/"+str(self.spp)+'psnrsntas.txt', 'a') as fp:
         fp.write("\n".join(str(item.item()) for item in self.psnrs))
         fp.write("\n")
    self.mses=[]
    self.psnrs=[]
    self.offset+=10
    return temp
    
def save(data,name):
    img= T.ToPILImage()(data)
    if img.mode != 'RGB':
     img = img.convert('RGB')
    img.save(name)
