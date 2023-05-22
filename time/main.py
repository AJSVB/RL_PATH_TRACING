import unet2
import unet1small
import unet1big
import env
import time
import ray.rllib.algorithms.appo as appo
import torch
import sys

config = {
    "framework": "torch",
    "vtrace_drop_last_ts": False,
    "optimizer": "adabelief",
    "use_critic": True,
    "use_gae": True,
    "use_kl_loss": True,
    "kl_coeff": 5e-7,
    "kl_target": 5e-8,
    "lambda": 0.2,
    "clip_param": 0.15,
    "lr": 1e-3,  
    "grad_clip": 4,
    "replay_buffer_num_slots": 40,  
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 1e-5,
    "normalize_actions": False,
    "train_batch_size": 4,  # Was 128
    "no_done_at_end": False,
    "num_envs_per_worker": 1,
    "num_workers": 1,   #Was 8
    "num_gpus": 4,
    "num_cpus_per_worker": 48,
    "num_gpus_per_worker": 4,
    "rollout_fragment_length": 4,
    "model":   {"custom_model": "UN"},
#    "evaluation_interval":2500,
#    "evaluation_duration": 100 
}

def train_ppo_model(
    spp=4, mode="", conf="111", interval=[700, 800]
):
    """Main code for our framework: Networks/agents get initialized and trained, possibly concurently

    Args:
        spp (int, optional): the spp count. Defaults to 4.
        mode (str, optional): the variant/baseline being run. Defaults to "".
        conf (str, optional): the scale of every of the 3 network, where 0 is the scaled down version,
            1 is the default, and 2 is the large version. Defaults to "111".
        interval (list, optional): the validation interval. Defaults to [700, 800].
    """
    trainours = mode == "vanilla" or mode == "notp" or mode == "notp1" or mode=="D" #all of our variants that require RL training
    spp = float(spp)
    algos = {}
    if trainours:
        def a(x):
         x.eval()
         return x
        def b(x):
         x.endeval()
         return x
        config["env_config"] = {
            "spp": spp,
            "mode": mode,
            "conf": conf,
            "interval": interval,
        }
        algos[mode] = appo.APPO(env=env.CustomEnv, config=config)
        for i in range(2500): #hardcoded number of iterations that corresponds to the wanted number of epochs
#            if i==config["evaluation_interval"]-1:
#             algos[mode].workers.foreach_env(a)
#             algos[mode].train()
#             algos[mode].workers.foreach_env(b)
#            else:
             algos[mode].train()
    else:
        modes = mode
        if not isinstance(modes,list):
            modes = [modes]
        sims = {}
        unets = {}
        for e, mode in enumerate(modes):
            sims[mode] = env.CustomEnv(
                {
                    "spp": spp,
                    "mode": mode,
                    "conf": conf,
                    "interval": interval,
                }
            )
            unet = unet2.UN(
                sims[mode].observation_space, sims[mode].action_space, None, None, None
            )
            unet.model_config = model
            unet.obs_space = sims[mode].observation_space
            unet.action_space = sims[mode].action_space
            unets[mode] = unet

        def f(x, mode):
            """returns the sampling importance recommendation

            Args:
                x (Tensor): the input of the sampling importance network
                mode (str): the mode description

            Returns:
                tensor: the sampling importance recommendation
            """            
            if "uni" in mode:
                return (torch.ones([1, 720, 720]).cuda(0) * spp).type(torch.int)
            else:
                b, _ = unets[mode](x, None, None)
                b = b.reshape(1, 720, 720).cuda(0)
                return b

        def g(x, e):
            """helper function to convert data

            Args:
                x (Tensor): the observation state
                e (int): On which gpu to transfer the data  

            Returns:
                _type_: a dictionary where the observation has the good type
            """            
            return {"obs": torch.Tensor(x).unsqueeze(0).cuda(e)}

        for i in range(2500):
            for e, mode in enumerate(modes):
                unets[mode] = unets[mode].cuda(e)
                a = sims[mode].reset()
                b = f(g(a, e), mode)
                for _ in range(20):
                    a, b, _, _ = sims[mode].step(b)
                    b = f(g(a, e), mode)
                unets[mode] = unets[mode].cpu()


"""
mode guide:
vanilla: our main method
grad: oursA1
uni: oursA2
notp1: oursB1
notp: oursB2
D: oursC
dasr: DASR
ntas: NTAS
imcduni: IMCD
"""
if __name__ == "__main__":
    te = sys.argv[-5:]
    spp, mode, conf, i1, i2 = te
    if conf[0] == "0":
        config["model"]["custom_model"] = "UNsmall"
    if conf[0] == "2":
        config["model"]["custom_model"] = "UNbig"

    train_ppo_model(spp, mode, conf, [int(i1), int(i2)])
