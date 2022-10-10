
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


def train_ppo_model():
    a = time.time()
    algo = appo.APPO(env=simulation.CustomEnv,config={
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.5,"denoising":False,"prob_sampling":True,"partition":[1]
            },
          'framework' :"torch",
"num_gpus":4,
"vf_loss_coeff":.4,
"momentum":.9,
"lr":1e-1,"lambda":.8,"kl_coeff":.6,"grad_clip":.400,"gamma":1,
"epsilon":0.4,
"entropy_coeff":1e-7,
"decay":.98,
"clip_param":.04, 
"train_batch_size":32,
#"num_envs_per_worker":1,
        'num_workers':4,
#"evaluation_num_workers":1,
'num_cpus_per_worker':3,
'num_gpus_per_worker':.16,
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
    for _ in range(500):
         algo.train()
    print(time.time()-a)
    # Save state of the trained Algorithm in a checkpoint.
   # algo.save("/tmp/rllib_checkpoint")
   # return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


#simulation.ground_truth("../datasets/temple/",name="truth.png")
if __name__ == "__main__":
    train_ppo_model()
