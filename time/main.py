import ray
import unet2
import env
import time
import ray.rllib.algorithms.appo as appo
import random
import random
import numpy as np
import math
import numpy.random as tune
import torch
c=3
model={
   "custom_model":"UN",
"conv_filters":[[16, [c, c], 1],[16, [c, c], 1],[2, [c, c], 1], [1, [c, c], 1]],
"fcnet_hiddens":[],
"post_fcnet_hiddens":[],
"post_fcnet_activation":"linear",
"vf_share_layers":False,
"no_final_linear":True}
config = {
          'framework' :"torch",
"num_gpus":4,
"vtrace_drop_last_ts":False,
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
"replay_buffer_num_slots":40, #Was 30
          "vf_loss_coeff": .5,
          "entropy_coeff": 1e-5 ,
"normalize_actions":False,
"train_batch_size":4, #Was 128
"num_envs_per_worker":1,
        'num_workers':1, #8, TODO
"num_gpus":4,
'num_cpus_per_worker':48,
'num_gpus_per_worker':4,
"rollout_fragment_length":4, 
  "model": model 
}




def train_ppo_model(spp=4,c=1,sppps=.5,i=1,mode=""):
  spp=float(spp)
  c = int(c)
  sppps = float(sppps)
  a = time.time()
  i=int(i)
  config["env_config"] = {'spp':spp, "sppps":sppps,"mode":mode}
 
  if mode=="notp" or mode=="":
    algo = appo.APPO(env=env.CustomEnv,config=config)
    from ray.air import session
    for _ in range(4000): 
         a = algo.train()
  else:
    sim = env.CustomEnv(config["env_config"])

    if "uni" in mode:
      f = lambda x :   (torch.ones([1,720,720]).cuda(0)*sppps).type(torch.int)
    else: 
     unet = unet2.UN(sim.observation_space,sim.action_space,None,None,None).cuda(0)
     unet.model_config= model
     unet.obs_space = sim.observation_space
     unet.action_space = sim.action_space
     def f(x):
      b,_ =  unet(x,None,None)
      b =b.reshape(1,720,720)
      return b

    def g(x):
      return {"obs":torch.Tensor(x).unsqueeze(0).cuda(0)}


    for i in range(4000):
      a=sim.reset()
      a=g(a)
      b=f(a)
      for j in range(20):
        a,b,_,_ = sim.step(b)
        a=g(a)
        b = f(a)




if __name__ == "__main__":
 import sys
 te = sys.argv[-2:]
 te, mode = te
 if mode=="vanilla":
  mode=""
 spp,c,sppps,i =  te,0,te,0
 i=int(i)
 train_ppo_model(spp,c,sppps,i,mode)
