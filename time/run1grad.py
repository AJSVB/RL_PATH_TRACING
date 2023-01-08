import ray
import simulation1grad 
import simulation1ntas
import simulation1dasr
import time
import unet2 as unet1
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.ddppo as ddppo
import torch
import ray.rllib.algorithms.appo as appo

import ray.rllib.algorithms.sac as sac #Needs a policy and a q model
import ray.rllib.algorithms.td3 as td3 #Gave a mistake (strnage)
import ray.rllib.algorithms.apex_ddpg as apex #Same mistake

import random

import random
import numpy as np
import math

import numpy.random as tune

def train_ppo_model(spp=4,c=1,sppps=.5,i=1,mode):
    spp=float(spp)
    c = int(c)
    sppps = float(sppps)
    a = time.time()
    i=int(i)

    if mode =="ntas":
     simulation1grad = simulation1ntas
    if mode == "dasr":
     simulation1grad = simulation1dasr

    env=simulation1grad.CustomEnv
    env_config={'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':spp, "sppps":sppps,"denoising":True,"prob_sampling":True,"partition":[1],"i":i
            }
    model_config = {
   "custom_model":"UN",
"conv_filters":[[16, [c, c], 1],[16, [c, c], 1],[2, [c, c], 1], [1, [c, c], 1]],
"fcnet_hiddens":[],
"post_fcnet_hiddens":[],
"post_fcnet_activation":"linear",
"vf_share_layers":False,
"no_final_linear":True,
}
    sim = env(env_config)
    unet = unet1.UN(sim.observation_space,sim.action_space,None,None,None).cuda(0)
    unet.model_config= model_config
    unet.obs_space = sim.observation_space
    unet.action_space = sim.action_space
    def f(x):
      b,_ =  unet(x,None,None)
      b =b.reshape(1,720,720)
      return b

    def g(x):
      return {"obs":torch.Tensor(x).unsqueeze(0).cuda(0)}


    sim = env(env_config)
    for i in range(2000):
      print(i)
      a=sim.reset()
      a=g(a)
      b=f(a)
      for j in range(20):
        a,b,_,_ = sim.step(b)
        a=g(a)
        b = f(a)
         


#simulation.ground_truth("../datasets/temple/",name="truth.png")
if __name__ == "__main__":
 import sys
 te = sys.argv[-2:]
 te, mode = te
 spp,c,sppps,i =  te,0,te,0
 i=int(i)
 train_ppo_model(spp,c,sppps,i,mode)

