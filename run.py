#import unet
import fcn
import simulation 
import ray
import time
#ray.init(num_gpus=4)

import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.ddppo as ddppo

import ray.rllib.algorithms.appo as appo
import random

def train_ppo_model():
    a = time.time()
    algo = appo.APPO(env=simulation.CustomEnv,config={
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.1,"denoising":True,
            },
          'framework' :"torch",
#"eager_tracing":True,
"clip_param":.45,"decay":.97,"epsilon":.01, "grad_clip":4,"lambda":.925,"lr":.0001, "momentum":.5,"vf_loss_coeff":.81,"entropy_coeff":1e-3,
"num_envs_per_worker":1,
        'num_workers':1,
#"evaluation_num_workers":1,
#'num_cpus_per_worker':10,
'num_gpus_per_worker':1,
"evaluation_interval":5,
"rollout_fragment_length":20, #Increase this
"train_batch_size":80,
"replay_buffer_num_slots":50,
#"disable_env_checking":True,
  "model":{
#"conv_activation":"tanh"
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
