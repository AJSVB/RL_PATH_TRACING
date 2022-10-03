
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

def generate_partition():
    cs=2 #math.e
    n = 1/sum([cs**(-i) for i in range(15)]) #TODO when 16 it segfaults
    return  [n*cs**(-i) for i in range(15)]



def train_ppo_model():
    a = time.time()
    algo = appo.APPO(env=simulation.CustomEnv,config={
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.1,"denoising":True,"prob_sampling":True, "partition":generate_partition()
            },
          'framework' :"torch",


"vf_loss_coeff":.3,"momentum":.7,"lr":.1,"lambda":.49,"kl_coeff":0,"grad_clip":400,"gamma":1,"epsilon":.8,"entropy_coeff":1e-5,"decay":.98,"clip_param":.1, #gamma was .47 but I believe 1 makes more sense



"num_envs_per_worker":2,
        'num_workers':1,
#"evaluation_num_workers":1,
#'num_cpus_per_worker':10,


'num_gpus_per_worker':4,
##"evaluation_interval":5,
"rollout_fragment_length":8, #was20
"train_batch_size":32,
#"replay_buffer_num_slots":100,
  "model":{
   "custom_model":"FCN"
}
})
    # Train for one iteration.
    for _ in range(1):
         algo.train()
    print(time.time()-a)
    # Save state of the trained Algorithm in a checkpoint.
   # algo.save("/tmp/rllib_checkpoint")
   # return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


#simulation.ground_truth("../datasets/temple/",name="truth.png")
train_ppo_model()
