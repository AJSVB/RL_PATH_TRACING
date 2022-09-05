import torch
import torchvision
from torch.nn import MSELoss
import gym
from gym import Env, spaces
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo

class PhysicSimulation:
    def __init__(self):
        self.HEIGHT = 720 
        self.WIDTH =  1280
        self.max = 2
        self.observations=torch.zeros((self.HEIGHT*self.WIDTH,3))
        self.reset()

    def reset(self):
        self.indexes = torch.zeros([self.HEIGHT, self.WIDTH], dtype = torch.int)
        self.indexes=self.indexes.view( -1, *self.indexes.shape[2:])
        self.count =0

    def simulate(self, x):
        x=x.flatten()
        x = x/np.sum(x)*len(x)
        x=np.round(x)
        #print(np.sum(x))
        #print(len(x))
        max = np.quantile(x,.5)
        idx = np.where(x>=max)[0]
        indexes = self.indexes[idx] 
        self.indexes[idx]= indexes+1
        self.count+=1


    def out(self,data):
        return data.view(self.HEIGHT,self.WIDTH,*data.shape[1:]).numpy()    

    def render(self):
        return self.out(self.observations)

    def observe(self):
        temp = np.concatenate((self.render(),self.out((self.indexes/self.max).unsqueeze(-1))),axis=-1)
        return temp




class Spec:
   def __init__(self,max_episode_steps):
    self.max_episode_steps = max_episode_steps
    self.id = "foo"


      





class CustomEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, env_config):
    super(CustomEnv, self).__init__()
    self.simulation = PhysicSimulation()
    self.WIDTH = self.simulation.WIDTH
    self.HEIGHT = self.simulation.HEIGHT
    self.action_space = spaces.Box(low=0,high=1, shape=(int(self.HEIGHT*self.WIDTH),))
    self.observation_space = spaces.Box(low=-1e-6, high=1, shape=
                    (self.HEIGHT,self.WIDTH,4), dtype=np.float32)
    self.spec = Spec(self.simulation.max)

  def step(self, action):
    self.simulation.simulate(action)
    observation = self.simulation.observe()
    reward = 0 
    done = self.spec.max_episode_steps <= self.simulation.count
#    print(np.isnan(observation).any())
    return observation,reward,done, {}

    
  def reset(self):
    self.simulation.reset()
    return self.simulation.observe()
    



ray.init(num_gpus=4)

import ray.rllib.algorithms.a3c as a3c

def train_ppo_model():
    algo = a3c.A3C(env=CustomEnv,config={
          'framework' :"torch",
"num_envs_per_worker":1,
        'num_workers':1,
'num_gpus_per_worker':4,
"evaluation_interval":1,
"rollout_fragment_length":10, 
"train_batch_size":40, 
#"sgd_minibatch_size":40,
  "model":{
    "conv_filters": [
        [16,[24,48], [21,36]],
        [32, [6, 6], 4],
        [256, [9, 9], 1],
    ]
    }
})
    algo.train()

train_ppo_model()
