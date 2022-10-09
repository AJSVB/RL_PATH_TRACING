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

def norm(a,denoising):
    if not denoising:
        return a*0
    return (a-torch.min(a))/(torch.max(a)-torch.min(a))


def ground_truth(path,number_images=10000,frame_number=1,HEIGHT=720,WIDTH=1280,name=""):
    dataset = get_ith_image(path,0,frame_number,HEIGHT,WIDTH) 
    for i in range(1,number_images):
      dataset = (dataset*i + get_ith_image(path,i,frame_number,HEIGHT,WIDTH) )/(i+1.)
    img= T.ToPILImage()(dataset.squeeze(0))
    img.save(path+name)

def get_truth(path,HEIGHT=480,WIDTH=640):
    image= Image.open(path)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x[:,:,-HEIGHT:,-WIDTH:].type(torch.float16).mean(1).unsqueeze(1)


def get_ith_image(path,i,frame_number = 1,HEIGHT=480,WIDTH=640):
    image = Image.open(path+str(frame_number).zfill(4) + "-" + str(i).zfill(5)+'.png0001.png')
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x[:,:,-HEIGHT:,-WIDTH:].type(torch.float16).mean(1).unsqueeze(1)

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
    return x.type(torch.float16).mean(0).unsqueeze(0)

def load_additional(path,frame_number=1,HEIGHT=480,WIDTH=640):
    dataset = torch.cat([get_add(path,a) for a in  ["Normal","DiffCol","Mist"]])
    dataset = dataset.permute([1,2,0])
    return dataset[-HEIGHT:,-WIDTH:]


def load_albedo(path,frame_number=1,HEIGHT=480,WIDTH=640):
    dataset = get_add(path,"DiffCol")
    dataset = dataset.permute([1,2,0])
    return dataset[-HEIGHT:,-WIDTH:]





class PhysicSimulation:

    def __init__(self,path,spp,frame_number=1, sppps=.1,list=None,add = None,albedo=None,HEIGHT=480,WIDTH=730,max=10,denoising=True,partition=None,CST=1):
        self.partition = partition
        self.CST=CST
        self.HEIGHT =  HEIGHT
        self.WIDTH =   WIDTH
        self.dataset = cached(list)
        self.sppps = sppps
        self.dataset=self.dataset.view( -1, *self.dataset.shape[2:])
        self.add = add
        self.albedo=albedo.reshape(-1,3)
        self.max = max
        self.reset()
        self.updated=False
        self.denoising=denoising
    def reset(self):
        self.permutation = torch.randperm(self.max)
        self.observations = self.dataset[:,:,self.permutation[0]] #self.albedo #torch.zeros(self.dataset.shape[:2])
        self.indexes = torch.ones([self.HEIGHT, self.WIDTH], dtype = torch.int) 
        self.indexes=self.indexes.view( -1, *self.indexes.shape[2:])
        self.variance = self.observations**2
        self.count = 1 #
        self.updated=False
    
    def sample(self,idx):
        indexes = self.indexes[idx] 
        self.indexes[idx]= indexes+1
        temp = self.dataset[idx,:,self.permutation[self.count]]
        self.observations[idx,:] = self.observations[idx,:] +  temp  
        self.variance[idx,:] = self.variance[idx,:] + temp**2 
        self.count+=1


    def simulate(self, x):
        x=x.flatten()
       # print(x)
        e=np.exp(x)
        s=np.rint(self.sppps*self.HEIGHT*self.WIDTH*e/np.sum(e)).astype(int)
        print(np.sum(s))
       # print(np.max(s))
       # print(np.min(s))

        idxs = []
        for i in range(min(10,np.max(s))):
          idxs.append(np.where(s>i)[0])
        idx=idxs[0]
        indexes = self.indexes[idx].unsqueeze(1)#.repeat(1,3)
        self.observations[idx,:]=self.observations[idx,:]*indexes
        self.variance[idx,:]=self.variance[idx,:]*indexes


        for x in idxs:
            self.sample(x)

        indexes = self.indexes[idx].unsqueeze(1) #.repeat(1,3)
        self.observations[idx,:]=self.observations[idx,:]/indexes
        self.variance[idx,:]=self.variance[idx,:]/indexes
        self.updated=False

        if random.random() > .9 and torch.sum(self.indexes)==2*self.observations.shape[0]:
            print(torch.var(self.indexes.float()))

    def out(self,data):
        return data.view(self.HEIGHT,self.WIDTH,*data.shape[1:]).type(torch.float16)    

    def render(self):        
      if self.denoising:
        if not self.updated:
    #      b=time.time()
          with FileLock('/home/ascardigli/RL_PATH_TRACING/tmp/0.pfm.lock'):
            self.denoised= torch.Tensor(denoising.denoise(self.out(self.observations),str(0)))
     #     print("lock included" + str(time.time()-b))
          self.updated=True
        return self.denoised
      else:
        return self.out(self.observations)


    def observe(self):
        a=self.render()
        rendersquared = self.observations**2
        temp = torch.cat((
self.out(self.observations),\
self.out(self.indexes/self.max).unsqueeze(-1), \
self.out((self.variance - rendersquared)),self.add, \
 norm((self.out(self.observations)-a),self.denoising) \
  ),axis=-1).permute(2,0,1)
        return temp


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
    loss=MS_SSIM(data_range=1,size_average=False,channel=1).cuda(gpu_id)
    return loss(d,e).cpu() 




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
    self.partition = env_config["partition"]
    self.CST= 10
    self.max = (int(self.spp/self.sppps) - int(1/self.sppps) ) *self.CST +1 #
    self.id = env_config.vector_index
    self.list = [get_ith_image(self.path,i,self.frame_number,self.HEIGHT,self.WIDTH) for i in range(self.max)]

    self.add = load_additional(self.path,1,self.HEIGHT,self.WIDTH)
    self.albedo = load_albedo(self.path,1,self.HEIGHT,self.WIDTH)

    self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps,self.list,self.add,self.albedo,self.HEIGHT,self.WIDTH,self.max,self.denoising,self.partition,self.CST)

    self.action_space = spaces.Box(low=-1e1,high=1e1,shape=(self.HEIGHT*self.WIDTH,))
    self.observation_space = spaces.Box(low=-1e-2, high=1, shape=
                    (7,self.HEIGHT,self.WIDTH), dtype=np.float16) #MACHINE PRECISION
    denoising.initialise("/home/ascardigli/datasets/temple/")

    self.spec = Spec(self.max)
    self.ground_truth = get_truth("/home/ascardigli/datasets/temple/"+"truth.png",self.HEIGHT,self.WIDTH)
    self.top = 0

  def step(self, action):
#    a=time.time()
    gd=self.ground_truth
    old = self.simulation.render()
    i=-0 #works with 0 outside of tune.py TODO
    old = MultiSSIM([old], [gd],i)[0]
    self.simulation.simulate(action)
 #   print("RT"+str(time.time()-a))
    observation = self.simulation.observe()
  #  print("observation"+str(time.time()-a))
    new = self.simulation.render()
    import ray
    new = MultiSSIM([new], [gd],i)[0]
#    print(time.time()-a)
    if random.random()>.9 and self.top<new:
        print(new)
        self.top = new
        if self.top>.973:
         self.insight()
    reward = - old + new
    done = self.spec.max_episode_steps <= self.simulation.count
#    print("wole loop" + str(time.time()-a)) 
  
    return observation.numpy(),reward.detach().numpy(),done, {}

  def insight(self): 
    img= self.simulation.indexes.unsqueeze(-1)
    norm = (img-torch.min(img))/(torch.max(img) - torch.min(img))
    te=str(random.random())
 #   print(te)
    save(self.simulation.out(norm),"/home/ascardigli/RL_PATH_TRACING/tmp/"+te+".png")

    
  def reset(self):
    self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps,self.list,self.add,self.albedo,self.HEIGHT,self.WIDTH,self.max,self.denoising,self.partition,self.CST)
    return self.simulation.observe().numpy()
    
  def render(self, mode='human', close=False):
    return self.simulation.render()

def save(data,name):
    img= T.ToPILImage()(data.permute([2,0,1]).type(torch.float32))
    img.save(name)
