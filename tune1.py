import fcn
import fcn1
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
        "TD3", 
        tune_config=tune.TuneConfig(
            metric="episode_reward_max",
            mode="max",
            search_alg=hyperopt_search,
            num_samples=100,time_budget_s=1e4,
max_concurrent_trials=1
        ),

        param_space={
            "env": simulation.CustomEnv,
'env_config':{'path': "/home/ascardigli/datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.5 ,"denoising":False , "prob_sampling":True,"partition":[1]},

          'framework' :"torch",
"num_gpus_per_worker":.2,
"num_workers":20,
#"num_gpus":3,
"num_cpus_per_worker":2,
"use_state_preprocessor":True,
"actor_hiddens": [],
"critic_hiddens":  [],
"min_sample_timesteps_per_iteration":100,
"replay_buffer_config":{
"capacity":400,
"learning_starts":400,
},
"model":{
"fcnet_hiddens":[],
"no_final_linear":True,
"custom_model":"FCN1"
},
"exploration_config":{
"random_timesteps":400,
"stddev":tune.choice([1,.3,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4]),
"final_scale":tune.choice([0,1])
},
            "gamma": tune.uniform(0,1),
"train_batch_size": tune.choice(range(4,24,10)),

            "target_noise":tune.uniform(.1,.9),
            "target_noise_clip":tune.uniform(.2,.8),
            "critic_lr": tune.choice([1e-1,3e-2,1e-2,3e-3,1e-4,3e-4,1e-4,3e-5, 1e-5,3e-5,1e-5,3e-6,1e-6]),
            "actor_lr":  tune.choice([1e-1,3e-2,1e-2,3e-3,1e-4,3e-4,1e-4,3e-5, 1e-5,3e-5,1e-5,3e-6,1e-6]),
            "tau":  tune.choice([1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-7,1e-7,3e-8,1e-8,1e-4,3e-5, 1e-5,3e-5,1e-5,3e-6,1e-6]),
            "lr": tune.choice([1e-1,3e-2,1e-2,3e-4,1e-3,3e-3,5e-4, 1e-4, 5e-5, 1e-5,1e-6]),
            "grad_clip": tune.choice([.1,.4,1,4,40,100]),
             "l2_reg":tune.choice([1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-7,1e-7,3e-8,1e-8,1e-4,3e-5, 1e-5,3e-5,1e-5,3e-6,1e-6]),
           
  }
   )
    results = tuner.fit()

    print("best hyperparameters: ", results.get_best_result().config)
