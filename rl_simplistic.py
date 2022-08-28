import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import time
import torchvision.transforms as T
import random


def get_ith_image(path,i,frame_number = 1):
    image = Image.open(path+str(frame_number).zfill(4) + "-" + str(i).zfill(5)+'.png')
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x

def aggregate_by_pixel(path,number_images,frame_number = 1):
    dataset = torch.cat([get_ith_image(path,i,frame_number) for i in range(number_images)],0)
    dataset = dataset.permute([2,3,1,0])
    return dataset


class PhysicSimulation:
    def __init__(self,path,number_images,frame_number=1):
        self.dataset=aggregate_by_pixel(path,number_images,frame_number)[-42:,-42:,:,:]
        self.ground_truth = self.dataset.mean(dim = 3)
        self.max = number_images
        self.HEIGHT = self.ground_truth.shape[0]
        self.WIDTH = self.ground_truth.shape[1]
        self.indexes = torch.zeros([self.HEIGHT, self.WIDTH], dtype = torch.int)
        self.observations= torch.zeros([self.HEIGHT,self.WIDTH,3])
        self.variance = torch.zeros([self.HEIGHT,self.WIDTH,3])
    def __len__(self):
        return self.max * self.ground_truth.shape[0] * self.ground_truth.shape[1]

    def simulate(self, x,y):
        i = self.indexes[x,y]
        self.indexes[x,y] =i+1
        out = self.dataset[x,y,:,i%self.max]
        self.observations[x,y,:] = (self.observations[x,y,:] * i +  out )/(i+1)
        self.variance[x,y,:] = (self.variance[x,y,:]*i + out**2 )/(i+1)
        #if i >= self.max:
        #    print("warning: number of precomputed samples is not big enough, information is redondant") 
       # return out
    
    def render(self):  
        return self.observations.numpy()
    
    def observe(self):
        render = self.render()
        temp = np.concatenate((render,np.expand_dims((self.indexes.numpy()/self.max),-1), self.variance.numpy() - render**2),axis=-1)           
#        print(np.min(temp))
        return temp

    def truth(self):
        return self.ground_truth.numpy()
    



class Spec:
   def __init__(self,max_episode_steps):
    self.max_episode_steps = max_episode_steps
    self.id = "3Drenderingenv"


import gym
from gym import Env, spaces
import numpy as np

class CustomEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, env_config):
    super(CustomEnv, self).__init__()
    print(env_config)
    self.path = env_config["path"]
    self.number_images = env_config["number_images"]
    self.frame_number = env_config["frame_number"]
    self.spp = env_config['spp']
    self.simulation = PhysicSimulation(self.path,self.number_images,self.frame_number)
    self.truth = self.simulation.truth()
    self.WIDTH = self.simulation.WIDTH
    self.HEIGHT = self.simulation.HEIGHT

    self.action_space = spaces.MultiDiscrete([self.HEIGHT,self.WIDTH])
    self.observation_space = spaces.Box(low=-1e-8, high=1, shape=
                    (self.HEIGHT,self.WIDTH,7), dtype=np.float32) #MACHINE PRECISION
    self.count = 0
    self.spec = Spec(self.WIDTH*self.HEIGHT*self.spp)
    
  def step(self, action):
    # Execute one time step within the environment
    self.simulation.simulate(*action)
    observation = self.simulation.observe()
    reward = - np.sum((observation[:,:,:3] - self.truth)**2)
    self.count+=1
    done = self.spec.max_episode_steps == self.count
    if not self.count%100:
        print(reward)
    return observation,reward,done, {}
    
  def reset(self):
    # Reset the state of the environment to an initial state
    self.simulation = PhysicSimulation(self.path,self.number_images,self.frame_number)
    return self.simulation.observe()
    
  def render(self, mode='human', close=False):
    return self.simulation.render()




import ray

ray.init(num_gpus=4)

import ray.rllib.algorithms.ppo as ppo
from ray import serve
def train_ppo_model():
                     
    algo = ppo.PPO(env=CustomEnv,config={
'env_config':{'path': "/scratch/datasets/Antoine/barcelona/",'number_images':100,'frame_number':1, 'spp':3
            },
          'framework' :"torch",
        'num_workers':4,
'num_gpus_per_worker':1,
   #     'conv_filters':[out_channels, kernel, stride]
    })
    
    # Train for one iteration.
    for _ in range(10):
         print(algo.train())
    print(algo.evaluate())
    # Save state of the trained Algorithm in a checkpoint.
    algo.save("/tmp/rllib_checkpoint")
    return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


checkpoint_path = train_ppo_model()
