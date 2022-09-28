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

#"clip_param":.45,"decay":.97,"epsilon":.01, "grad_clip":4,"lambda":.925,"lr":.0001, "momentum":.5,"vf_loss_coeff":.81,"entropy_coeff":1e-3,
"num_envs_per_worker":2,
        'num_workers':1,
#"evaluation_num_workers":1,
#'num_cpus_per_worker':10,
'num_gpus_per_worker':4,
"evaluation_interval":5,
"rollout_fragment_length":20, #was20
"train_batch_size":20,
#"replay_buffer_num_slots":60,
  "model":{
   "custom_model":"FCN"
}
})
    # Train for one iteration.
    for _ in range(30):
         algo.train()
    print(time.time()-a)
    # Save state of the trained Algorithm in a checkpoint.
   # algo.save("/tmp/rllib_checkpoint")
   # return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"


#simulation.ground_truth("../datasets/temple/",name="truth.png")
train_ppo_model()
