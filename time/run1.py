import ray
import simulation1 
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


def train_ppo_model(spp=4,c=1,sppps=.5,i=1):
    spp=int(spp)
    c = int(c)
    sppps = float(sppps)
    a = time.time()
    i=int(i)
    algo = appo.APPO(env=simulation1.CustomEnv,config={
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':spp, "sppps":sppps,"denoising":True,"prob_sampling":True,"partition":[1],"i":i
            },
          'framework' :"torch",
"num_gpus":4,

"vtrace_drop_last_ts":False,

#"vf_loss_coeff":.05,"momentum":.9,"lr":.1,"lambda":.8,"kl_coeff":1,"grad_clip":.400,"gamma":1,
#"kl_target":1,"epsilon":0.01,"entropy_coeff":1e-3,"decay":.98,"clip_param":.04, 


#  "clip_param": 0.9726497874484089,   
#"decay": 0.97,
#, "entropy_coeff": 1e-7,
#   "epsilon": 0.21244584717938497,   "grad_clip": 4000.0,   "kl_coeff": 0.49175633009791462,
#  "kl_target": 0.4022912297489361,   "lambda": 0.07157041286837718,   "lr": 1e-2,  "momentum": 0.9,
 # "use_critic": False,  "use_gae": True,   "use_kl_loss": True,  
#   "vf_loss_coeff": 0.3323018458897308,
   "optimizer": "adabelief", 



            "use_critic":True,
            "use_gae":True,
            "use_kl_loss":True,
            "kl_coeff": 5e-7,
            "kl_target": 5e-8,
            "lambda": .2,
            "clip_param": .15,
            "lr": 1e-3, #tune.choice([1e-5,3e-5,1e-6,3e-6]) ,
            "grad_clip": 4,
          #"momentum": tune.uniform(.5,1) ,
          #"epsilon": .3,
"replay_buffer_num_slots":60,
          "vf_loss_coeff": .5,
          "entropy_coeff": 1e-5 ,

#"sgd_minibatch_size":16,


#"gamma":0,
"normalize_actions":False,
"train_batch_size":4, #Was 128
"num_envs_per_worker":1,
        'num_workers':1, #8, TODO
#"",
"num_gpus":4,
#"evaluation_num_workers":1,
'num_cpus_per_worker':48,
'num_gpus_per_worker':4,
#"evaluation_interval":10,
"rollout_fragment_length":4, 
  "model":{
   "custom_model":"UN",
"conv_filters":[[16, [c, c], 1],[16, [c, c], 1],[2, [c, c], 1], [1, [c, c], 1]],
"fcnet_hiddens":[],
"post_fcnet_hiddens":[],
"post_fcnet_activation":"linear",
"vf_share_layers":False,
"no_final_linear":True,
}
})
    # Train for one iteration.
    from ray.air import session
    for _ in range(10000):
         a = algo.train()
    # Save state of the trained Algorithm in a checkpoint.
   # algo.save("/tmp/rllib_checkpoint")
   # return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"

#simulation.ground_truth("../datasets/temple/",name="truth.png")
if __name__ == "__main__":
 import sys
 te = sys.argv[-4:]
 spp,c,sppps,i =  te
 i=2**int(i)
 train_ppo_model(spp,c,sppps,i)
