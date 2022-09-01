	


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
   return math.floor(number/strid) +1

def get_ith_image(path,i,frame_number = 1,HEIGHT=480,WIDTH=640):
    image = Image.open(path+str(frame_number).zfill(4) + "-" + str(i).zfill(5)+'.png')
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x[:,:,-HEIGHT:,-WIDTH:]

def aggregate_by_pixel(path,number_images,frame_number = 1,HEIGHT=480,WIDTH=640):
    dataset = torch.cat([get_ith_image(path,i,frame_number,HEIGHT,WIDTH) for i in range(number_images)],0)
    dataset = dataset.permute([2,3,1,0])
    return dataset


class PhysicSimulation:
    def __init__(self,path,spp,frame_number=1, sppps=.1):
        self.HEIGHT = 480 #TODO optimize!
        self.WIDTH =  640
        self.max = int(spp/sppps)
#        print(self.max)
        self.sppps = sppps
        self.dataset=aggregate_by_pixel(path,self.max,frame_number,self.HEIGHT,self.WIDTH)
        self.ground_truth = self.dataset.mean(dim = 3).numpy()
        self.dataset=self.dataset.view( -1, *self.dataset.shape[2:])
        self.indexes = torch.zeros([self.HEIGHT, self.WIDTH], dtype = torch.int)
        self.indexes=self.indexes.view( -1, *self.indexes.shape[2:])
        self.observations= torch.zeros([self.HEIGHT,self.WIDTH,3])
        self.observations=self.observations.view( -1, *self.observations.shape[2:])
        self.variance = torch.zeros([self.HEIGHT,self.WIDTH,3])
        self.variance=self.variance.view( -1, *self.variance.shape[2:])
        self.count = 0

    def reset(self):
        self.indexes = torch.zeros([self.HEIGHT, self.WIDTH], dtype = torch.int)
        self.indexes=self.indexes.view( -1, *self.indexes.shape[2:])
        self.observations= torch.zeros([self.HEIGHT,self.WIDTH,3])
        self.observations=self.observations.view( -1, *self.observations.shape[2:])
        self.variance = torch.zeros([self.HEIGHT,self.WIDTH,3])
        self.variance=self.variance.view( -1, *self.variance.shape[2:])
        self.count = 0



    def __len__(self):
        return self.max * self.HEIGHT *self.WIDTH

    def simulate(self, x):
        x=x.flatten()
        #print(len(x))
        max = np.percentile(x,100-100*self.sppps)
        idx = np.where(x>=max)[0]
        #print(len(idx))
        indexes = self.indexes[idx] 
        self.indexes[idx]= indexes+1
        temp = self.dataset[idx,:,self.count]
        #print(indexes)
        indexes = indexes.unsqueeze(1).repeat(1,3)
        #print(indexes)
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
        return temp

    def truth(self):
        return self.ground_truth
    



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
    self.frame_number = env_config["frame_number"]
    self.spp = env_config['spp']
    self.sppps = env_config["sppps"]
    self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps)
    self.number_images = self.simulation.max #=Horizon
    self.truth = self.simulation.truth()
    self.WIDTH = self.simulation.WIDTH
    self.HEIGHT = self.simulation.HEIGHT
    self.action_space = spaces.Box(low=0,high=1, shape=(self.HEIGHT*self.WIDTH,))
    self.observation_space = spaces.Box(low=-1e-6, high=1, shape=
                    (self.HEIGHT,self.WIDTH,7), dtype=np.float32) #MACHINE PRECISION
    self.count = 0
    self.spec = Spec(self.number_images)

  def step(self, action):
    old = np.sum((self.simulation.render() - self.truth)**2)
    self.simulation.simulate(action)
    observation = self.simulation.observe()
    #print(np.min(observation))
    #print(np.max(observation))
    #print(old)
    #print(np.sum((observation[:,:,:3] - self.truth)**2))
    reward = (old - np.sum((observation[:,:,:3] - self.truth)**2))
    self.count+=1
    done = self.spec.max_episode_steps <= self.count
#    print(reward)
    #self.total = self.total + reward
    #print(np.max(observation))
    #print(np.min(observation))
    #print(observation.shape)
    return observation,reward,done, {}

    
  def reset(self):
    # Reset the state of the environment to an initial state
    self.count =0
    #save(self.simulation.render(),"render.png")
    self.simulation.reset()
    return self.simulation.observe()
    
  def render(self, mode='human', close=False):
    return self.simulation.render()

def save(data,name):
    img= T.ToPILImage()(torch.Tensor(data).permute([2,0,1]))
    img.save(name)


import ray

ray.init(num_gpus=4)

import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.ddppo as ddppo

from ray import serve
def train_ppo_model():
    a = time.time()
    algo = ppo.PPO(env=CustomEnv,config={
'env_config':{'path': "/scratch/datasets/Antoine/barcelona/",'number_images':None,\
'frame_number':1, 'spp':4, "sppps":.1
            },
          'framework' :"torch",
#"eager_tracing":True,


        'num_workers':9,
#"evaluation_num_workers":1,
'num_gpus_per_worker':.44,
"evaluation_interval":-1,
#"rollout_fragment_length":40,
"train_batch_size":120,
"sgd_minibatch_size":120
#"batch_mode":"complete_episodes"
   #     'conv_filters':[out_channels, kernel, stride]
    })
    # Train for one iteration.
    for _ in range(1):
         print(algo.train())
    print(time.time()-a)
    # Save state of the trained Algorithm in a checkpoint.
    algo.save("/tmp/rllib_checkpoint")
    return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


checkpoint_path = train_ppo_model()
