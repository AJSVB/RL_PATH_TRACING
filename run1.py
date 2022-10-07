
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
    algo = td3.TD3(env=simulation.CustomEnv,config={
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.5,"denoising":True,"prob_sampling":True,"partition":[1],
            },
          'framework' :"torch",
#"num_cpus_for_driver":46,
"num_gpus":4,
"use_state_preprocessor":True,
"actor_hiddens": [],
"critic_hiddens":  [],
"min_sample_timesteps_per_iteration":20,
"replay_buffer_config":{
"capacity":1000,
"learning_starts":20,
},
"model":{
"fcnet_hiddens":[],
"no_final_linear":True,
"custom_model":"FCN1"
},
"exploration_config":{
"random_timesteps":200,
"stddev":1e-1
},
            "gamma": 1,
"train_batch_size":4,
            "target_noise":.4,
            "target_noise_clip":.6,
            "critic_lr":  2e-5,
            "actor_lr": 2e-5,
            "tau": 1e-7,


            "lr": 5e-4,
            "grad_clip": 40,

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
