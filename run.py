
import fcn
import ray
import simulation 
import time

import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.ddppo as ddppo

import ray.rllib.algorithms.appo as appo

import ray.rllib.algorithms.sac as sac #Needs a policy and a q model
import ray.rllib.algorithms.td3 as td3 #Gave a mistake (strnage)
import ray.rllib.algorithms.apex_ddpg as apex #Same mistake

import random

import random
import numpy as np
import math
def generate_partition():
    le = round(random.random()*10)
    l = []
    r=1
    for i in range(le-1):
        t=random.random()*r
        r=r-t
        l.append(t)
    l.append(r)
    return np.sort(l)[::-1]

import numpy.random as tune
def train_ppo_model():
    a = time.time()
    algo = appo.APPO(env=simulation.CustomEnv,config={
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.1,"denoising":False,"prob_sampling":True,"partition":[1]
            },
          'framework' :"torch",
"num_gpus":4,
#"num_envs_per_worker":1,
        'num_workers':8,
#"evaluation_num_workers":1,
'num_cpus_per_worker':6,
'num_gpus_per_worker':.5,
"rollout_fragment_length":4, #was20
#"replay_buffer_num_slots":30,
  "model":{
   "custom_model":"UN",
"fcnet_hiddens":[],
"no_final_linear":True,
},

"train_batch_size":4 ,
            "gamma": 1,
            "kl_coeff": .6 ,
            "lambda": .2 ,
            "clip_param": .05 ,
            "lr": .1 ,
            "grad_clip": 40 ,
          "decay": .99,
          "momentum": .3,
          "epsilon": .05 ,
          "vf_loss_coeff":.6, #tune.uniform(0,1),
          "entropy_coeff": 1e-4 #tune.choice([1e-6,1e-7,1e-8,1e-9,1e-5,1e-4,1e-3,1e-2,1e-1])





})
    # Train for one iteration.
    for _ in range(500):
         algo.train()
    print(time.time()-a)
    # Save state of the trained Algorithm in a checkpoint.
   # algo.save("/tmp/rllib_checkpoint")
   # return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


#simulation.ground_truth("../datasets/temple/",name="truth.png")
if __name__ == "__main__":
    train_ppo_model()
