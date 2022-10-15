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

import numpy.random as tune

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


def train_ppo_model():
    a = time.time()
    algo = appo.APPO(env=simulation.CustomEnv,config={
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.5,"denoising":False,"prob_sampling":True,"partition":[1]
            },
          'framework' :"torch",
"num_gpus":4,



#"vf_loss_coeff":.05,"momentum":.9,"lr":.1,"lambda":.8,"kl_coeff":1,"grad_clip":.400,"gamma":1,
#"kl_target":1,"epsilon":0.01,"entropy_coeff":1e-3,"decay":.98,"clip_param":.04, 


#  "clip_param": 0.9726497874484089,   
"decay": 0.97,
#, "entropy_coeff": 1e-7,
#   "epsilon": 0.21244584717938497,   "grad_clip": 4000.0,   "kl_coeff": 0.49175633009791462,
#  "kl_target": 0.4022912297489361,   "lambda": 0.07157041286837718,   "lr": 1e-2,  "momentum": 0.9,
  "use_critic": False,  "use_gae": True,   "use_kl_loss": True,  
#   "vf_loss_coeff": 0.3323018458897308,
   "optimizer":"adabelief", 



#            "use_critic":tune.choice([True,False]),
#            "use_gae":tune.choice([True,False]),
#            "use_kl_loss":tune.choice([True,False]),
            "kl_coeff": 5e-3,
            "kl_target": 5e-4,
            "lambda": .2,
            "clip_param": .15,
            "lr": 1e-5 ,
            "grad_clip": 4,
          "momentum": .9 ,
          "epsilon": .5,
          "vf_loss_coeff": .5,
          "entropy_coeff": 1e-5 ,



        "vtrace_drop_last_ts" : False,

"gamma":0,
"normalize_actions":False,
"train_batch_size":8,
#"num_envs_per_worker":1,
        'num_workers':8,

#"",
#"evaluation_num_workers":1,
'num_cpus_per_worker':6,
'num_gpus_per_worker':.5,
"evaluation_interval":10,
"rollout_fragment_length":2, #was20
#"replay_buffer_num_slots":30,
  "model":{
   "custom_model":"FCN",
"fcnet_hiddens":[],
"no_final_linear":True,

}
})
    # Train for one iteration.
    for _ in range(30):
         a = algo.train()['episode_reward_max']
    print(time.time()-a)
    # Save state of the trained Algorithm in a checkpoint.
   # algo.save("/tmp/rllib_checkpoint")
   # return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


#simulation.ground_truth("../datasets/temple/",name="truth.png")
if __name__ == "__main__":
    train_ppo_model()
