

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
    image = Image.open(path+str(frame_number).zfill(4) + "-" + str(i).zfill(5)+'.png0001.png')
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x[:,:,-HEIGHT:,-WIDTH:]

def aggregate_by_pixel(path,number_images,frame_number = 1,HEIGHT=480,WIDTH=640):
    dataset = torch.cat([get_ith_image(path,i,frame_number,HEIGHT,WIDTH) for i in range(number_images)],0)
    dataset = dataset.permute([2,3,1,0])
    return dataset


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
    def __init__(self,path,spp,frame_number=1, sppps=.1):
        self.HEIGHT = 84 # 720  #TODO optimize!
        self.WIDTH =  84 # 1280
        self.max = int(spp/sppps)
#        print(self.max)
        self.sppps = sppps
        self.dataset=aggregate_by_pixel(path,self.max,frame_number,self.HEIGHT,self.WIDTH)
        self.ground_truth = self.dataset.mean(dim = 3).numpy()
        self.dataset=self.dataset.view( -1, *self.dataset.shape[2:])
        self.add = load_additional(path,1,self.HEIGHT,self.WIDTH)
        self.reset()

    def reset(self):
        self.permutation = torch.randperm(self.max)
        self.observations= self.dataset[:,:,self.permutation[0]]
        self.indexes = torch.ones([self.HEIGHT, self.WIDTH], dtype = torch.int)
        self.indexes=self.indexes.view( -1, *self.indexes.shape[2:])
        self.variance = self.observations**2
        self.count = int(1/self.sppps)


    def __len__(self):
        return self.max * self.HEIGHT *self.WIDTH

    def simulate(self, x):
        x=x.flatten()
        #print(len(x))
        max = np.quantile(x,1-self.sppps)
        idx = np.where(x>=max)[0]
        #print(len(idx))
        indexes = self.indexes[idx] 
        self.indexes[idx]= indexes+1
        temp = self.dataset[idx,:,self.permutation[self.count]]
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
        return np.concatenate((temp,self.add),axis=-1)

    def truth(self):
        return self.ground_truth
    



class Spec:
   def __init__(self,max_episode_steps):
    self.max_episode_steps = max_episode_steps
    self.id = "3Drenderingenv"

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
def MultiSSIM(a,b):
    return ssim(torch.Tensor(a).permute([2,0,1]).unsqueeze(0),torch.Tensor(b).permute([2,0,1]).unsqueeze(0),data_range=1)



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
    self.action_space = spaces.Box(low=0,high=1, shape=(int(self.HEIGHT*self.WIDTH),))
    self.observation_space = spaces.Box(low=-1e-6, high=1, shape=
                    (self.HEIGHT,self.WIDTH,21), dtype=np.float32) #MACHINE PRECISION
    self.spec = Spec(self.number_images)

  def step(self, action):
    old = MultiSSIM(self.simulation.render(), self.truth)
    self.simulation.simulate(action)
    observation = self.simulation.observe()
    #print(np.min(observation))
    #print(np.max(observation))
    #print(old)
    #print(np.sum((observation[:,:,:3] - self.truth)**2))
    reward = - old + MultiSSIM(observation[:,:,:3], self.truth)
    done = self.spec.max_episode_steps <= self.simulation.count
#    print(reward)
    #self.total = self.total + reward
    #print(np.max(observation))
    #print(np.min(observation))
    #print(observation.shape)
    #print((np.isnan(observation) | np.isnan(observation)).any())
    #print(reward)
    return observation,reward.detach().numpy(),done, {}

    
  def reset(self):
    # Reset the state of the environment to an initial state
    self.count =0
    #save(self.simulation.render(),"render.png")
    if random.random()>.9:
        img= self.simulation.indexes.unsqueeze(-1)
        norm = (img-torch.min(img))/(torch.max(img) - torch.min(img))
        save(self.simulation.out(norm),str(random.random())+".png")
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

import ray.rllib.algorithms.appo as appo
import ray.rllib.algorithms.a3c as a3c
import ray.rllib.algorithms.a2c as a2c
import ray.rllib.algorithms.ars as ars
import ray.rllib.algorithms.bc as bc
import ray.rllib.algorithms.crr as crr
import ray.rllib.algorithms.ddpg as ddpg
import ray.rllib.algorithms.es as es
import ray.rllib.algorithms.pg as pg

from ray import serve
def train_ppo_model():
    a = time.time()
    algo = ddppo.DDPPO(env=CustomEnv,config={
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.1
            },
          'framework' :"torch",
#"eager_tracing":True,

"num_envs_per_worker":1,
        'num_workers':4,

#"entropy_coeff":1e-6,
#"evaluation_num_workers":1,
'num_gpus_per_worker':1,
"evaluation_interval":1,
#"rollout_fragment_length":10, #Increase this
#"train_batch_size":10,
#"grad_clip":4,
#"sgd_minibatch_size":40,
#"vf_clip_param":10000
#"batch_mode":"complete_episodes"
#  "model":{
#    "conv_filters": [
##        [8 , [6,6] , [3,4]],
##        [16, [18, 24], [7, 9]], #
#        [16,[24,48], [21,36]],
#        [32, [6, 6], 4],
#        [256, [9, 9], 1],
#    ]
#    }
})
    # Train for one iteration.
    for _ in range(100):
         algo.train()
    print(time.time()-a)
    # Save state of the trained Algorithm in a checkpoint.
   # algo.save("/tmp/rllib_checkpoint")
   # return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


train_ppo_model()



"""
"dim":720,
"vf_share_layers": True,
  "conv_filters": [
     [16,[7,7],[2,3]],
     [16, [3,3], 2], 
     [32, [3, 3], 2], 
     [48, [3, 3], 2], 
     [64, [3, 3], 2],
     [96, [3, 3], 2], 
     [128, [3, 3], 2], 
     [192, [3, 3], 2],
     [256, [3, 3], 2] 
]},
"""
