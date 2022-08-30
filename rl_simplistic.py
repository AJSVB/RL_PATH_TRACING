


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


def stride(number,strid):
   return math.floor(number/strid)

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
    def __init__(self,path,number_images,frame_number=1, batch_rad = 8, strid =1):
        self.stride = strid
        self.dataset=aggregate_by_pixel(path,number_images,frame_number)[-10:,-10:,:,:]
        self.ground_truth = self.dataset.mean(dim = 3)
        self.max = number_images
        self.HEIGHT = self.ground_truth.shape[0]
        self.WIDTH = self.ground_truth.shape[1]
        self.indexes = torch.zeros([self.HEIGHT, self.WIDTH], dtype = torch.int)
        self.observations= torch.zeros([self.HEIGHT,self.WIDTH,3])
        self.variance = torch.zeros([self.HEIGHT,self.WIDTH,3])
        self.batch_ray_rad = batch_rad
    def __len__(self):
        return self.max * self.ground_truth.shape[0] * self.ground_truth.shape[1]

    def simulate(self, x,y):
        x=x*self.stride+self.batch_ray_rad
        y=y*self.stride+self.batch_ray_rad
        i = self.indexes[x-self.batch_ray_rad:x+self.batch_ray_rad,y-self.batch_ray_rad:y+self.batch_ray_rad]
        self.indexes[x-self.batch_ray_rad:x+self.batch_ray_rad,y-self.batch_ray_rad:y+self.batch_ray_rad] =i+1
        i=i.repeat(3,1,1).permute(1,2,0)
        temp = self.dataset[x-self.batch_ray_rad:x+self.batch_ray_rad,y-self.batch_ray_rad:y+self.batch_ray_rad,:,:]
        inde = (i%self.max).long().unsqueeze(-1)
        out = temp.gather(-1,inde).squeeze(-1)

        self.observations[x-self.batch_ray_rad:x+self.batch_ray_rad,y-self.batch_ray_rad:y+self.batch_ray_rad,:] = (self.observations[x-self.batch_ray_rad:x+self.batch_ray_rad,y-self.batch_ray_rad:y+self.batch_ray_rad,:] * i +  out )/(i+1)
        self.variance[x-self.batch_ray_rad:x+self.batch_ray_rad,y-self.batch_ray_rad:y+self.batch_ray_rad,:] = (self.variance[x-self.batch_ray_rad:x+self.batch_ray_rad,y-self.batch_ray_rad:y+self.batch_ray_rad,:]*i + out**2 )/(i+1)
        
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
    self.path = env_config["path"]
    self.number_images = env_config["number_images"]
    self.frame_number = env_config["frame_number"]
    self.spp = env_config['spp']
    self.batch_rad = env_config["batch_rad"]
    self.stride = env_config["stride"]
    self.simulation = PhysicSimulation(self.path,self.number_images,self.frame_number,self.batch_rad,self.stride)
    self.truth = self.simulation.truth()
    self.WIDTH = self.simulation.WIDTH
    self.HEIGHT = self.simulation.HEIGHT

    self.maxW=stride(self.WIDTH-2*self.batch_rad+self.stride,self.stride)
    self.maxH=stride(self.HEIGHT-2*self.batch_rad+self.stride,self.stride)

    self.action_space = spaces.MultiDiscrete([self.maxH,self.maxW])
    self.observation_space = spaces.Box(low=-1e-8, high=1, shape=
                    (self.HEIGHT,self.WIDTH,7), dtype=np.float32) #MACHINE PRECISION
    self.count = 0
    self.spec = Spec(math.ceil(self.WIDTH*self.HEIGHT*self.spp/((2*self.batch_rad+1)**2)))
    self.total=0    

  def step(self, action):
    old = np.sum((self.simulation.render() - self.truth)**2)
    self.simulation.simulate(*action)
    observation = self.simulation.observe()
    reward = (old - np.sum((observation[:,:,:3] - self.truth)**2))
    self.count+=1
    done = self.spec.max_episode_steps <= self.count
    #print(reward)
    #self.total = self.total + reward
    #print(np.max(observation))
    #print(np.min(observation))
    #print(observation.shape)
    return observation,reward,done, {}

    
  def reset(self):
    # Reset the state of the environment to an initial state
    self.count =0
    self.total = 0
    print(self.simulation.observe()[:,:,3])
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
'env_config':{'path': "/scratch/datasets/Antoine/barcelona/",'number_images':16,\
'frame_number':1, 'spp':4, "batch_rad":2, "stride":2
            },
          'framework' :"torch",
        'num_workers':2,
'num_gpus_per_worker':1,
"evaluation_interval":1,
#"rollout_fragment_length":111,
"train_batch_size":400,
#"batch_mode":"complete_episodes"
   #     'conv_filters':[out_channels, kernel, stride]
    })
    
    # Train for one iteration.
    for _ in range(10):
         print(algo.train())
    # Save state of the trained Algorithm in a checkpoint.
    algo.save("/tmp/rllib_checkpoint")
    return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


checkpoint_path = train_ppo_model()
