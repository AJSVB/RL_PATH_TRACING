import fcn
import fcn1
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

def train_ppo_model():
    a = time.time()
#    algo = apex.ApexDDPG(env=simulation.CustomEnv,config={
    algo = td3.TD3(env=simulation.CustomEnv,config={
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.5,"denoising":False,"prob_sampling":True,"partition":[1],
            },
          'framework' :"torch",

"num_gpus_per_worker":.12,
"num_workers":22,
"num_gpus":4,
"num_cpus_per_worker":2,
"use_state_preprocessor":True,
"actor_hiddens": [],
"critic_hiddens":  [],
"min_sample_timesteps_per_iteration":100,
"replay_buffer_config":{
"capacity":400,
"learning_starts":400,
},
"model":{
"fcnet_hiddens":[],
"no_final_linear":True,
"custom_model":"FCN1"
},
"exploration_config":{
"random_timesteps":400,
"stddev":1e-3,
#"final_scale":0
},
            "gamma": 0.27,
"train_batch_size":4,
            "target_noise": .27,
            "target_noise_clip":.8,
            "critic_lr":  1e-2,
            "actor_lr": 3e-5,
            "tau": 3e-2,
            "l2_reg":3e-5,

            "lr": 1e-2,
            "grad_clip": 1,

  }
   )

    # Train for one iteration.
    for _ in range(1000):
         algo.train()
    print(time.time()-a)
    # Save state of the trained Algorithm in a checkpoint.
   # algo.save("/tmp/rllib_checkpoint")
   # return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


#simulation.ground_truth("../datasets/temple/",name="truth.png")

train_ppo_model()
