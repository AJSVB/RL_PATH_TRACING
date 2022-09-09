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

import unet

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
    dataset = torch.cat([get_add(path,a) for a in  ["Normal"]])
    dataset = dataset.permute([1,2,0])
    return dataset[-HEIGHT:,-WIDTH:]

class PhysicSimulation:
    def __init__(self,path,spp,frame_number=1, sppps=.1,list=None,add = None,HEIGHT=480,WIDTH=730,max=10):
        self.HEIGHT =  HEIGHT
        self.WIDTH =   WIDTH
        self.dataset = cached(list)
#        self.dataset=aggregate_by_pixel(path,self.max,frame_number,self.HEIGHT,self.WIDTH)
        self.sppps = sppps
        self.dataset=self.dataset.view( -1, *self.dataset.shape[2:])
        self.add = add
        self.max = max
        self.reset()

    def reset(self):
        self.permutation = torch.randperm(self.max)
        self.observations = torch.zeros(self.dataset.shape[:2]) #self.dataset[:,:,self.permutation[0]]
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
        temp = np.concatenate((self.out(self.observations.mean(-1).unsqueeze(-1)),self.out((self.indexes/self.max).unsqueeze(-1)), self.out((self.variance - rendersquared).mean(-1).unsqueeze(-1))),axis=-1)           
        return np.concatenate((temp,self.add),axis=-1)

    def truth(self):
        return np.mean(self.out(self.dataset),-1)


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
    self.HEIGHT = 720 
    self.WIDTH =   1280
    self.max = int(self.spp/self.sppps) #- int(1/sppps)+1
    self.list = [get_ith_image(self.path,i,self.frame_number,self.HEIGHT,self.WIDTH) for i in range(self.max)]
    self.add = load_additional(self.path,1,self.HEIGHT,self.WIDTH)
    self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps,self.list,self.add,self.HEIGHT,self.WIDTH,self.max)
    self.action_space = spaces.Box(low=0,high=1,shape=(self.HEIGHT*self.WIDTH,))
    self.observation_space = spaces.Box(low=-1e-6, high=1, shape=
                    (self.HEIGHT,self.WIDTH,6), dtype=np.float32) #MACHINE PRECISION
    self.spec = Spec(self.max)
    self.ground_truth = self.simulation.truth()
    self.top = 0

  def step(self, action):
    old = MultiSSIM(self.simulation.render(), self.ground_truth)
    self.simulation.simulate(action)
    observation = self.simulation.observe()
    new = MultiSSIM(self.simulation.render(), self.ground_truth)
    #if self.top<new:
    #    print(new)
    #    self.top = new
    reward = - old + new
    done = self.spec.max_episode_steps <= self.simulation.count
    
    return observation,reward.detach().numpy(),done, {}

    
  def reset(self):
    if random.random()>.99:
        img= self.simulation.indexes.unsqueeze(-1)
        norm = (img-torch.min(img))/(torch.max(img) - torch.min(img))
        save(self.simulation.out(norm),str(random.random())+".png")
    self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps,self.list,self.add,self.HEIGHT,self.WIDTH,self.max)
    return self.simulation.observe()
    
  def render(self, mode='human', close=False):
    return self.simulation.render()

def save(data,name):
    img= T.ToPILImage()(torch.Tensor(data).permute([2,0,1]))
    img.save(name)


import numpy as np
from typing import Dict, List
import gym

import ray

#ray.init(num_gpus=4)

import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.ddppo as ddppo

import ray.rllib.algorithms.appo as appo
import random

from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining

if True:
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=3,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.9),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "grad_clip": [.4,4,40,400],
	  "decay": lambda:random.uniform(.95,1),
          "momentum": [0,.1,.3,.5,.7,.9,.99],
          "epsilon": [0.01,0.1,0.3],
          "vf_loss_coeff":lambda: random.uniform(0,1),
          "entropy_coeff": [1e-5,1e-4,1e-3,1e-2,1e-1]
        }
    )
    
    tuner = tune.Tuner(
        "APPO", 
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=pbt,
            num_samples=10,
        ),
        param_space={
            "env": CustomEnv,
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.1 },
          'framework' :"torch",
        'num_workers':4,
"entropy_coeff":1e-3,
'num_gpus_per_worker':1,
#"evaluation_interval":5,
"rollout_fragment_length":8, #Increase this
"train_batch_size":32,
"replay_buffer_num_slots":50,
  "model":{
   "custom_model":"UN"
}
  }
   )
    results = tuner.fit()

    print("best hyperparameters: ", results.get_best_result().config)
