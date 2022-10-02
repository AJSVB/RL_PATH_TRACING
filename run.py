
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

def train_ppo_model():
    a = time.time()
    algo = appo.APPO(env=simulation.CustomEnv,config={
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.1,"denoising":True,"prob_sampling":True,
            },
          'framework' :"torch",


"vf_loss_coeff":.3,"momentum":.7,"lr":.1,"lambda":.49,"kl_coeff":0,"grad_clip":400,"gamma":1,"epsilon":.4,"entropy_coeff":1e-5,"decay":.98,"clip_param":.1, #gamma was .47 but I believe 1 makes more sense



"num_envs_per_worker":2,
        'num_workers':1,
#"evaluation_num_workers":1,
#'num_cpus_per_worker':10,


'num_gpus_per_worker':1,
##"evaluation_interval":5,
"rollout_fragment_length":8, #was20
"train_batch_size":32,
#"replay_buffer_num_slots":100,
  "model":{
   "custom_model":"UN"
}
})
    # Train for one iteration.
    for _ in range(10):
         algo.train()
    print(time.time()-a)
    # Save state of the trained Algorithm in a checkpoint.
   # algo.save("/tmp/rllib_checkpoint")
   # return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


#simulation.ground_truth("../datasets/temple/",name="truth.png")
train_ppo_model()
