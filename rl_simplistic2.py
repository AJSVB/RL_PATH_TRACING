	


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


def aggregate_by_pixel(number_images,HEIGHT=480,WIDTH=640):
    return torch.zeros((HEIGHT,WIDTH,3,number_images))


class PhysicSimulation:
    def __init__(self):
        self.HEIGHT = 720 
        self.WIDTH =  1280
        self.max = 30
        self.dataset=aggregate_by_pixel(self.max,self.HEIGHT,self.WIDTH)
        self.ground_truth = self.dataset.mean(dim = 3).numpy()
        self.dataset=self.dataset.view( -1, *self.dataset.shape[2:])
        self.reset()

    def reset(self):
        self.observations= self.dataset[:,:,0]
        self.indexes = torch.zeros([self.HEIGHT, self.WIDTH], dtype = torch.int)
        self.indexes=self.indexes.view( -1, *self.indexes.shape[2:])
        self.count =0


    def simulate(self, x):
        x=x.flatten()
        max = np.quantile(x,.5)
        idx = np.where(x>=max)[0]
        indexes = self.indexes[idx] 
        self.indexes[idx]= indexes+1
        temp = self.dataset[idx,:,self.count]
        indexes = indexes.unsqueeze(1).repeat(1,3)
        self.observations[idx,:] =  temp
        self.count+=1


    def out(self,data):
        return data.view(self.HEIGHT,self.WIDTH,*data.shape[1:]).numpy()    

    def render(self):
        return self.out(self.observations)

    def observe(self):
        temp = np.concatenate((self.render(),self.out((self.indexes/self.max).unsqueeze(-1))),axis=-1)
        return temp


    def truth(self):
        return self.ground_truth
    



class Spec:
   def __init__(self,max_episode_steps):
    self.max_episode_steps = max_episode_steps
    self.id = "foo"

from torch.nn import MSELoss
       
def MSE(a,b):
    return MSELoss()(torch.Tensor(a),torch.Tensor(b))



import gym
from gym import Env, spaces
import numpy as np

class CustomEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, env_config):
    super(CustomEnv, self).__init__()
    self.simulation = PhysicSimulation()
    self.number_images = self.simulation.max #=Horizon
    self.truth = self.simulation.truth()
    self.WIDTH = self.simulation.WIDTH
    self.HEIGHT = self.simulation.HEIGHT
    self.action_space = spaces.Box(low=0,high=1, shape=(int(self.HEIGHT*self.WIDTH),))
    self.observation_space = spaces.Box(low=-1e-6, high=1, shape=
                    (self.HEIGHT,self.WIDTH,4), dtype=np.float32)
    self.spec = Spec(self.number_images)

  def step(self, action):
    self.simulation.simulate(action)
    observation = self.simulation.observe()
    reward = 0 
    done = self.spec.max_episode_steps <= self.simulation.count
    return observation,reward,done, {}

    
  def reset(self):
    self.simulation.reset()
    return self.simulation.observe()
    

import ray

ray.init(num_gpus=4)

import ray.rllib.algorithms.ppo as ppo
def train_ppo_model():
    algo = ppo.PPO(env=CustomEnv,config={
          'framework' :"torch",
"num_envs_per_worker":1,
        'num_workers':4,
'num_gpus_per_worker':1,
"evaluation_interval":1,
"rollout_fragment_length":10, 
"train_batch_size":40, 
"sgd_minibatch_size":40,
  "model":{
"vf_share_layers":True,
    "conv_filters": [
        [16,[24,48], [21,36]],
        [32, [6, 6], 4],
        [256, [9, 9], 1],
    ]
    }
})
    for _ in range(100):
         algo.train()

train_ppo_model()
