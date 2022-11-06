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

L=1

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
    return x[:,:,-HEIGHT:,-WIDTH:]#.type(torch.float16) #.mean(1).unsqueeze(1)


def get_ith_image(path,i,frame_number = 1,HEIGHT=480,WIDTH=640):
    image = Image.open(path+str(frame_number).zfill(4) + "-" + str(i).zfill(5)+'.png0001.png')
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x[:,:,-HEIGHT:,-WIDTH:]#.type(torch.float16) #.mean(1).unsqueeze(1)

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
    return x #.mean(0).unsqueeze(0)#.type(torch.float16)

def load_additional(path,frame_number=1,HEIGHT=480,WIDTH=640):
    dataset = torch.cat([get_add(path,a) for a in  ["Normal","DiffCol","Mist"]])
    dataset = dataset.permute([1,2,0])
    return dataset[-HEIGHT:,-WIDTH:]


def load_albedo(path,frame_number=1,HEIGHT=480,WIDTH=640):
    dataset = get_add(path,"DiffCol")
    dataset = dataset.permute([1,2,0])
    return dataset[-HEIGHT:,-WIDTH:]





class PhysicSimulation:

    def __init__(self,path,spp,frame_number=1, sppps=.1,HEIGHT=480,WIDTH=730,max=10,denoising=True,partition=None,CST=1):

        self.model,self.data = train.main_worker()
        self.model.eval()
        self.dataset = np.transpose(self.data.data(0),(1,2,3,0))[:720,:720]
        self.partition = partition
        self.add, self.gd = self.data.get(0)
        self.add = self.add.permute(1,2,0).unsqueeze(-1).to(dtype=torch.float32)
        self.gd=self.gd.to(dtype=torch.float32)
        self.CST=CST
        self.HEIGHT =  HEIGHT
        self.WIDTH =   WIDTH
        self.sppps = sppps
        self.shape=self.dataset.shape
        self.dataset=self.dataset.reshape( -1,*self.dataset.shape[2:])
        self.max = max
        self.reset()
        self.updated=False
        self.denoising=denoising
        self.a=torch.Tensor(range(720)).unsqueeze(0).repeat(720,1).unsqueeze(-1)/720.
        self.b=torch.Tensor(range(720)).unsqueeze(1).repeat(1,720).unsqueeze(-1)/720.
    def reset(self):
        self.permutation = torch.randperm(self.max)
        self.observations = -1 * torch.ones([self.HEIGHT*self.WIDTH,3,8])
        self.indexes = torch.ones([self.HEIGHT, self.WIDTH], dtype = torch.float32)*1e-8 # 
        self.indexes=self.indexes.view( -1, *self.indexes.shape[2:])
        self.variance = self.observations**2
        self.count = 0 #
        self.updated=False
    
    def sample(self,idx):
        indexes = self.indexes[idx] 
        self.indexes[idx]= indexes+1
        temp = self.dataset[idx,:,self.permutation[self.count]]
        self.observations[idx,:] = self.observations[idx,:] +  temp  
        self.variance[idx,:] = self.variance[idx,:] + temp**2 
        self.count+=1


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
        if random.random()>.99:
            print(x)
            print(s)
            print(np.sum(s))
        self.observations = self.data.generate(self.dataset,s)
        self.updated=False


    def out(self,data):
        return data.view(self.HEIGHT,self.WIDTH,-1)#.type(torch.float16)    

    def render(self):        
      if not self.updated:
        with torch.no_grad():
          self.denoised= self.model(self.data.__getitem__(0,self.dataset.reshape(self.shape)))
      self.updated=True
      return self.denoised.reshape(3,-1).permute(1,0)


    def observe(self):
        a=self.render()
#        rendersquared = self.observations**2
        temp = torch.cat((
self.out(self.observations),\
#self.out(self.indexes/torch.max(self.indexes)), \
#self.out((self.variance - rendersquared)),\
self.out(self.add), \
self.out(a), \
  self.a,self.b \
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
    print(d.shape)
    print(e.shape)
    return loss(d,e.unsqueeze(0)).cpu() 




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
    self.WIDTH =   720
    self.partition = env_config["partition"]
    self.CST= 10
    self.max = int((self.spp)/self.sppps)  *self.CST  # #
    self.id = env_config.vector_index

    self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps,self.HEIGHT,self.WIDTH,self.max,self.denoising,self.partition,self.CST)

    self.action_space = spaces.Box(low=0,high=1,shape=(int(self.HEIGHT*self.WIDTH/L/L),))
    self.observation_space = spaces.Box(low=-1, high=1, shape=
                    (int(self.HEIGHT/L),int(self.WIDTH/L),71), dtype=np.float32) #MACHINE PRECISION
    self.spec = Spec(self.max)
    self.counter= 0
    self.top=0

  def step(self, action):
    a,b,c,d=self.crop()
#    gd= self.ground_truth[:,:,a:b,c:d]
    old = self.simulation.out(self.simulation.render())
    i=-0 #works with 0 outside of tune.py TODO
    if L!=1:
        action=action.reshape(int(self.HEIGHT/L),int(self.WIDTH/L))
        x= np.zeros((int(self.HEIGHT/L),int(self.WIDTH/L)))
        y= np.concatenate((x,x),0)
        if self.counter==0:
            action = np.concatenate((action,x),0)
            action = np.concatenate((action,y),1)
        if self.counter==1:
            action = np.concatenate((x,action),0)
            action = np.concatenate((action,y),1)
        if self.counter==2:
            action = np.concatenate((action,x),0)
            action = np.concatenate((y,action),1)
        if self.counter==3:
            action = np.concatenate((x,action),0)
            action = np.concatenate((y,action),1)
        action = action.reshape(-1)
    
    self.simulation.simulate(action)
    observation,gd = self.simulation.observe()
    new = self.simulation.out(self.simulation.render())
    print(old.shape)
    print(gd.shape)
    old = MultiSSIM([old], [gd],i)[0]
    new = MultiSSIM([new], [gd],i)[0]
    if  self.top<new:
        print(new)
        self.top = new
        if self.top>.9806:
         self.insight()
    reward = 10**(new)
    done = self.spec.max_episode_steps <= self.simulation.count
    return observation.numpy().transpose(1,2,0),reward.detach().numpy(),done, {}

  def insight(self): 
    img= self.simulation.indexes.unsqueeze(-1)
    norm = (img-torch.min(img))/(torch.max(img) - torch.min(img))
    te=str(self.top.item())
    save(self.simulation.out(norm),"/home/ascardigli/RL_PATH_TRACING/tmp/"+te+".png")


  def crop(self):
    if L==1:
        return 0,self.HEIGHT,0,self.WIDTH
    counter=self.counter
    a=int(self.HEIGHT/L)
    b=int(self.WIDTH/L)
    i=(counter%2)==1
    j=counter>1
    return a*i,a*(i+1),b*j,b*(j+1)
    print(x.shape)
    return x[:,a*i:a*(i+1),b*j:b*(j+1)]
    
  def reset(self):
    self.counter+=4
    if self.counter==4:
        self.counter=0
        self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps,self.HEIGHT,self.WIDTH,self.max,self.denoising,self.partition,self.CST)
    a,b,c,d=self.crop()
    temp ,_= self.simulation.observe()
    return temp.numpy()[:,a:b,c:d].transpose(1,2,0)
    
  def render(self, mode='human', close=False):
    return self.simulation.render()

def save(data,name):
    img= T.ToPILImage()(data.permute([2,0,1]).type(torch.float32))
    img.save(name)
