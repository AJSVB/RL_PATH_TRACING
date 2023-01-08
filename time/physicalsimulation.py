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
import numpy as np
from torch.autograd import Function
from torch.autograd import Variable
import random
from torchvision.transforms import *

from torchvision.transforms import functional as F


def nostate(st):
  return "notp" in st or "dasr"==st

L=1
psnr = PeakSignalNoiseRatio().cuda(0)


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
        if sim.mode=="grad":
         return observations
        else:
         return obs


    @staticmethod
    def backward(ctx, dL_dout):
        dS_dn = ctx.saved_tensors[0]
        if len(dL_dout.shape)==3:
         dL_dout=dL_dout.unsqueeze(0)
        dL_din = torch.sum(torch.mul(dL_dout ,Variable(dS_dn)), dim=1, keepdim=True)
        return tuple([dL_din] + [None]*7)






def p(x,y):
   a=1
#   print(x+str(y))

#L1 = torch.nn.L1Loss()

list = []

import random
from torchvision.transforms import *

from torchvision.transforms import functional as F

size=(720,720)
scale=(0.08, 1.0)
ratio=(3.0 / 4.0, 4.0 / 3.0)
interpolation=InterpolationMode.BILINEAR

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

    def __init__(self,spp, sppps=.1,HEIGHT=480,WIDTH=730,sel=None,offset=0,mode=""):
        self.model,self.data,self.criterion,self.optimizer,self.scheduler = \
sel.model,sel.data,sel.criterion,sel.optimizer,sel.scheduler
        self.mode = mode
        self.HEIGHT =  HEIGHT
        self.WIDTH =   WIDTH
        self.sppps = sppps
        self.number = 1200
        self.offset = offset
        self.reset()
        self.new(-1)
        self.shape= [8,720,720]
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
        if self.mode=="ntas":
         self.state = -1 * torch.ones([3,self.HEIGHT,self.WIDTH]).cuda(0)
        else:
         self.state = -1 * torch.ones([32,self.HEIGHT,self.WIDTH]).cuda(0)
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
     if 'grad' in self.mode or "dasr" in self.mode or "ntas" in self.mode:
       self.observations = Render.apply(x,self) 
     else:

        if not "uni" in self.mode:
         x=torch.Tensor(x).cuda(0)
         x = x - torch.min(x) 
         x=torch.flatten(x).type(torch.float64)
         N= torch.sum(x)
         temp = self.sppps*self.WIDTH*self.HEIGHT 
         x=x*temp/N
         N=temp
         s= self.round_retain_sum(x,N)
        else:
         s=x
        if random.random()>.999:
            print(x.cpu())
            print(s.cpu())
            print(torch.sum(s).cpu())
        s[s<0]=-1
        s[s>8] = 8
        self.s=s
        self.observations = self.data.generate(self.dataset,s,self.count) 
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
          if nostate(self.mode):
           input= torch.cat((m1,m2),0).unsqueeze(0)
          else:
           input= torch.cat((m1,m2,m3),0).unsqueeze(0)

          self.denoised, self.state= self.model(input)
          if self.mode=="ntas":
           self.state = self.denoised
          loss = self.criterion(self.denoised, self.gd.unsqueeze(0)) 
          if not self.inval(): #self.offset<800 or self.offset>=900:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
          self.denoised = torch.clip(self.denoised.detach(),0,1)
          self.state = torch.clip(self.state.detach(),-1,1)
          if random.random()< 0.0001:
           t = str(self.offset+self.count-1)
           print(t)
           plt.imshow(m1.cpu().mean(0))
           plt.savefig("images/"+t+"obs.png")
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
           plt.imshow(self.denoised[0].to(torch.float).permute(1,2,0).detach().cpu())
           plt.savefig("images/"+t+"out.png")
           plt.clf()
           plt.imshow(self.s.detach().cpu().reshape(720,720))
           plt.savefig("images/"+t+"hitmap.png")

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
        if nostate(self.mode):
         input = m2
        else:
         input= torch.cat((m2,m3),0)
        self.count+=1
        return input.cpu(), self.gd


