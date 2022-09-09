import unet
import simulation 

import ray

#ray.init(num_gpus=4)

import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.ddppo as ddppo

import ray.rllib.algorithms.appo as appo
import random

from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining

if True:

    space={
            "lambda": hp.uniform("height",0.9, 1.0),
            "clip_param": hp.uniform("clip_param",0.01, 0.9),
            "lr": hp.choice("lr",[1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
            "grad_clip": hp.choice("grad_clip",[.4,4,40,400]),
	  "decay": hp.uniform("decay",.95,1),
          "momentum": hp.choice("momentum",[0,.1,.3,.5,.7,.9,.99]),
          "epsilon": hp.choice("epsilon",[0.01,0.1,0.3]),
          "vf_loss_coeff": hp.uniform("vf_loss_coeff",0,1),
          "entropy_coeff": hp.choice("entropy_coeff",[1e-5,1e-4,1e-3,1e-2,1e-1])
        }

    hyperopt_search = HyperOptSearch(space,
        metric="episode_reward_mean", mode="max",
        points_to_evaluate=current_best_params)



    scheduler = ASHAScheduler(
        metric="episode_reward_mean",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        "APPO", 
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            max_concurrent=1,
            search_alg=hyperopt_search,
            scheduler=scheduler,
            num_samples=50,
        ),
        param_space={
            "env": CustomEnv,
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.1 },
          'framework' :"torch",
        'num_workers':4,
'num_gpus_per_worker':1,
#"evaluation_interval":5,
"rollout_fragment_length":20, #Increase this
"train_batch_size":80,
"replay_buffer_num_slots":50,
  "model":{
   "custom_model":"FCN"
}
  }
   )
    results = tuner.fit()

    print("best hyperparameters: ", results.get_best_result().config)
