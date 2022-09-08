import torch
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
    return x

def load_additional(path,frame_number=1,HEIGHT=480,WIDTH=640):
    dataset = torch.cat([get_add(path,a) for a in  ["DiffCol","UV","IndexMA","Position","Normal","Mist"]])
    dataset = dataset.permute([1,2,0])
    return dataset[-HEIGHT:,-WIDTH:]

class PhysicSimulation:
    def __init__(self,path,spp,frame_number=1, sppps=.1,list=None,add = None):
        self.HEIGHT =  2000 
        self.WIDTH =   2000
        self.max = int(spp/sppps) #- int(1/sppps)+1
        self.dataset = cached(list)
#        self.dataset=aggregate_by_pixel(path,self.max,frame_number,self.HEIGHT,self.WIDTH)
        self.sppps = sppps
        self.ground_truth = self.dataset.mean(dim = 3).numpy()
#        self.dataset=self.dataset.view( -1, *self.dataset.shape[2:])
        self.add = add
        self.reset()

    def reset(self):
        self.permutation = torch.randperm(self.max)
        self.observations = torch.zeros([*self.dataset.shape[:2]]) #self.dataset[:,:,self.permutation[0]]
        self.indexes = torch.ones([self.HEIGHT, self.WIDTH], dtype = torch.int)
        self.indexes=self.indexes.view( -1, *self.indexes.shape[2:])
        self.variance = self.observations**2
        self.count = 0 #1
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


    def out(self,data):
        return data.view(self.HEIGHT,self.WIDTH,*data.shape[1:]).numpy()    

    def render(self):
        return self.out(self.observations)


    def observe(self):
        rendersquared = self.observations**2
        temp = np.concatenate((self.render(),self.out((self.indexes/self.max).unsqueeze(-1)), self.out(self.variance - rendersquared)),axis=-1)           
        return np.concatenate((temp,self.add),axis=-1)

    def truth(self):
        return self.ground_truth
    



class Spec:
   def __init__(self,max_episode_steps):
    self.max_episode_steps = max_episode_steps
    self.id = "3Drenderingenv"

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
def MultiSSIM(a,b):
    return ms_ssim(torch.Tensor(a).permute([2,0,1]).unsqueeze(0),torch.Tensor(b).permute([2,0,1]).unsqueeze(0),data_range=1)



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
    self.HEIGHT = 2000
    self.WIDTH =   2000
    self.max = int(self.spp/self.sppps) #- int(1/self.sppps)+1
    self.list = [get_ith_image(self.path,i,self.frame_number,self.HEIGHT,self.WIDTH) for i in range(self.max)]
    self.add = load_additional(self.path,1,self.HEIGHT,self.WIDTH)

    self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps,self.list,self.add)
    self.number_images = self.simulation.max #=Horizon
    self.truth = self.simulation.truth()
    self.WIDTH = self.simulation.WIDTH
    self.HEIGHT = self.simulation.HEIGHT
    self.action_space = spaces.Box(low=0,high=1,shape=(self.HEIGHT*self.WIDTH,))
    self.observation_space = spaces.Box(low=-1e-6, high=1, shape=
                    (self.HEIGHT,self.WIDTH,21), dtype=np.float32) #MACHINE PRECISION
    self.spec = Spec(self.number_images)

  def step(self, action):
    old = MultiSSIM(self.simulation.render(), self.truth)
    self.simulation.simulate(action)
    observation = self.simulation.observe()
    #print(old)
    reward = - old + MultiSSIM(self.simulation.render(), self.truth)
    done = self.spec.max_episode_steps <= self.simulation.count
    return observation,reward.detach().numpy(),done, {}

    
  def reset(self):
    if random.random()>.99:
        img= self.simulation.indexes.unsqueeze(-1)
        norm = (img-torch.min(img))/(torch.max(img) - torch.min(img))
        save(self.simulation.out(norm),str(random.random())+".png")
    self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps,self.list,self.add)
    return self.simulation.observe()
    
  def render(self, mode='human', close=False):
    return self.simulation.render()

def save(data,name):
    img= T.ToPILImage()(torch.Tensor(data).permute([2,0,1]))
    img.save(name)


def data_to_image(data):
    img= T.ToPILImage()(torch.Tensor(data).permute([2,0,1]))
    plt.imshow(np.asarray(img))
    plt.show()
    
def image_to_image(data):
    img= T.ToPILImage()(torch.Tensor(data))
    plt.imshow(np.asarray(img))
    plt.show()
    
def save(data,name):
    img= T.ToPILImage()(torch.Tensor(data))
    img.save(name)
def average(dataset,number):
    sampling = (torch.rand(dataset.shape[3])*dataset.shape[3]).long()[:number]
    print(sampling)
    print( dataset[:,:,:,sampling].shape)
    temp = dataset[:,:,:,sampling]
    if number == 0:
        temp = torch.zeros(dataset[:,:,:,0].squeeze(-1).shape)
    elif number ==1:
        temp = temp.squeeze(-1)
    elif number != 1:
        temp = temp.mean(dim = 3)
    return temp
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#from torch_geometric import ssim

def MSE(a,b):
    return np.sum((a.numpy()-b.numpy())**2)
def SSIM(a,b):
    return ssim(torch.Tensor(a),b,data_range=1).item()
def MultiSSIM(a,b):
    return ms_ssim(torch.Tensor(a),torch.Tensor(b),data_range=1).item()

import random
env = CustomEnv({'path': "../datasets/temple/",'frame_number':1, 'spp':1000, "sppps":1})


mse_a = []
ssim_a = []
mssim_a = []


def f(i):
    return average(env.simulation.dataset,i)

truth = torch.tensor(env.simulation.truth()).permute([2,0,1]).unsqueeze(0)

for i in range(0,20):
    print(i)
    temp = f(i).permute([2,0,1]).unsqueeze(0)
    mse_a.append(MSE(truth,temp))
    ssim_a.append(SSIM(truth,temp))
    mssim_a.append(MultiSSIM(truth,temp))

    
print(truth.squeeze().shape)
save(truth.squeeze(),"truth.png")

import matplotlib.pyplot as plt
plt.plot(mse_a)
plt.savefig("mse")
plt.clf()
plt.plot(ssim_a)
plt.savefig("ssim")
plt.clf()
plt.plot(mssim_a)
plt.savefig("msssim")
print(mse_a)
print(ssim_a)
print(mssim_a)

