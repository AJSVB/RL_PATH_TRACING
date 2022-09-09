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
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=3,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.9),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "grad_clip": [.4,4,40,400],
	  "decay": lambda:random.uniform(.95,1),
          "momentum": [0,.1,.3,.5,.7,.9,.99],
          "epsilon": [0.01,0.1,0.3],
          "vf_loss_coeff":lambda: random.uniform(0,1),
          "entropy_coeff": [1e-5,1e-4,1e-3,1e-2,1e-1]
        }
    )
    
    tuner = tune.Tuner(
        "APPO", 
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=pbt,
            num_samples=10,
        ),
        param_space={
            "env": CustomEnv,
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.1 },
          'framework' :"torch",
        'num_workers':4,
"entropy_coeff":1e-3,
'num_gpus_per_worker':1,
#"evaluation_interval":5,
"rollout_fragment_length":8, #Increase this
"train_batch_size":32,
"replay_buffer_num_slots":50,
  "model":{
   "custom_model":"UN"
}
  }
   )
    results = tuner.fit()

    print("best hyperparameters: ", results.get_best_result().config)
