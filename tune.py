import fcn
import simulation 
import run
import ray

#ray.init(num_gpus=4)

import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.ddppo as ddppo

import ray.rllib.algorithms.appo as appo
import random

from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
import hyperopt as hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
if True:
    ray.init(num_gpus=4,num_cpus=48)
    hyperopt_search = HyperOptSearch()

    tuner = tune.Tuner(
        "APPO", 
        tune_config=tune.TuneConfig(
            metric="episode_reward_max",
            mode="max",
            search_alg=hyperopt_search,
            num_samples=1,time_budget_s=1e4,
max_concurrent_trials=1
        ),
        param_space={
            "env": simulation.CustomEnv,
'env_config':{'path': "/home/ascardigli/datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.1 ,"denoising":False , "prob_sampling":True,"partition":[1]},

          'framework' :"torch",

"num_gpus":3,
'num_workers':15,
#"evaluation_num_workers":1,
'num_cpus_per_worker':3,
'num_gpus_per_worker':.16,
"evaluation_interval":10,
"rollout_fragment_length":2, 

"train_batch_size":tune.choice(range(4,32,4)),
  "model":{
   "custom_model":"FCN",
},
            "gamma": 1,
            "kl_coeff":tune.uniform(0.,1.),
            "lambda": tune.uniform(0., 1.0),
            "clip_param": tune.uniform(0.001, 0.99),
            "lr": tune.choice([1e-1,1e-2,1e-3, 5e-4, 1e-4, 5e-5, 1e-5,1e-6,1e-8]),
            "grad_clip": tune.choice([.04,.4,4,40,400,4000]),
          "decay": tune.uniform(.95,1),
          "momentum": tune.choice([0,.1,.3,.5,.7,.9,.99]),
          "epsilon": tune.uniform(0.,1.),
          "vf_loss_coeff": tune.uniform(0,1),
          "entropy_coeff": tune.choice([1e-6,1e-7,1e-8,1e-9,1e-5,1e-4,1e-3,1e-2,1e-1])

  }
   )
    results = tuner.fit()

    print("best hyperparameters: ", results.get_best_result().config)
