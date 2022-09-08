import torch
import math
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import time
import torchvision.transforms as T
import random

def get_ith_image(path,i,frame_number = 1,HEIGHT=480,WIDTH=640):
    image = Image.open(path+str(frame_number).zfill(4) + "-" + str(i).zfill(5)+'.png0001.png')
    x = TF.to_tensor(image)
    x.unsqueeze_(0)
    return x[:,:,-HEIGHT:,-WIDTH:]

def aggregate_by_pixel(path,number_images,frame_number = 1,HEIGHT=480,WIDTH=640):
    dataset = torch.cat([get_ith_image(path,i,frame_number,HEIGHT,WIDTH) for i in range(number_images)],0)
    dataset = dataset.permute([2,3,1,0])
    return dataset

def cached(list):
    return torch.cat(list,0).permute([2,3,1,0])


def get_add(path,detail):
    image = Image.open(path+"~"+detail+'0004.png')
    x = TF.to_tensor(image)
    if x[0,:].equal(x[1,:]):
        x = x[0:1,:]
    return x

def load_additional(path,frame_number=1,HEIGHT=480,WIDTH=640):
    dataset = torch.cat([get_add(path,a) for a in  ["Normal"]])
    dataset = dataset.permute([1,2,0])
    return dataset[-HEIGHT:,-WIDTH:]

class PhysicSimulation:
    def __init__(self,path,spp,frame_number=1, sppps=.1,list=None,add = None,HEIGHT=480,WIDTH=730,max=10):
        self.HEIGHT =  HEIGHT
        self.WIDTH =   WIDTH
        self.dataset = cached(list)
#        self.dataset=aggregate_by_pixel(path,self.max,frame_number,self.HEIGHT,self.WIDTH)
        self.sppps = sppps
        self.dataset=self.dataset.view( -1, *self.dataset.shape[2:])
        self.add = add
        self.max = max
        self.reset()

    def reset(self):
        self.permutation = torch.randperm(self.max)
        self.observations = torch.zeros(self.dataset.shape[:2]) #self.dataset[:,:,self.permutation[0]]
        self.indexes = torch.ones([self.HEIGHT, self.WIDTH], dtype = torch.int)
        self.indexes=self.indexes.view( -1, *self.indexes.shape[2:])
        self.variance = self.observations**2
        self.count = 0 #1

    def simulate(self, x):
        x=x.flatten()
        max = np.quantile(x,1-self.sppps)
        idx = np.where(x>=max)[0]
        indexes = self.indexes[idx] 
        self.indexes[idx]= indexes+1
        temp = self.dataset[idx,:,self.permutation[self.count]]
        indexes = indexes.unsqueeze(1).repeat(1,3)
        self.observations[idx,:] = (self.observations[idx,:]*indexes +  temp )/(indexes+1)
        self.variance[idx,:] = (self.variance[idx,:]*indexes + temp**2 )/(indexes+1)
        self.count+=1

    def out(self,data):
        return data.view(self.HEIGHT,self.WIDTH,*data.shape[1:]).numpy()    

    def render(self):
        return self.out(self.observations)


    def observe(self):
        rendersquared = self.observations**2
        temp = np.concatenate((self.render(),self.out((self.indexes/self.max).unsqueeze(-1)), self.out((self.variance - rendersquared).mean(-1).unsqueeze(-1))),axis=-1)           
        return np.concatenate((temp,self.add),axis=-1)

    def truth(self):
        return np.mean(self.out(self.dataset),-1)


class Spec:
   def __init__(self,max_episode_steps):
    self.max_episode_steps = max_episode_steps
    self.id = "3Drenderingenv"

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
def MultiSSIM(a,b):
    return ms_ssim(torch.Tensor(a).permute([2,0,1]).unsqueeze(0),torch.Tensor(b).permute([2,0,1]).unsqueeze(0),data_range=1)



import gym
from gym import Env, spaces
import numpy as np

class CustomEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, env_config):
    super(CustomEnv, self).__init__()
    self.path = env_config["path"]
    self.frame_number = env_config["frame_number"]
    self.spp = env_config['spp']
    self.sppps = env_config["sppps"]
    self.HEIGHT = 720 
    self.WIDTH =   1280
    self.max = int(self.spp/self.sppps) #- int(1/sppps)+1
    self.list = [get_ith_image(self.path,i,self.frame_number,self.HEIGHT,self.WIDTH) for i in range(self.max)]
    self.add = load_additional(self.path,1,self.HEIGHT,self.WIDTH)
    self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps,self.list,self.add,self.HEIGHT,self.WIDTH,self.max)
    self.action_space = spaces.Box(low=0,high=1,shape=(self.HEIGHT*self.WIDTH,))
    self.observation_space = spaces.Box(low=-1e-6, high=1, shape=
                    (self.HEIGHT,self.WIDTH,8), dtype=np.float32) #MACHINE PRECISION
    self.spec = Spec(self.max)
    self.ground_truth = self.simulation.truth()
    self.top = 0

  def step(self, action):
    old = MultiSSIM(self.simulation.render(), self.ground_truth)
    self.simulation.simulate(action)
    observation = self.simulation.observe()
    new = MultiSSIM(self.simulation.render(), self.ground_truth)
    #if self.top<new:
    #    print(new)
    #    self.top = new
    reward = - old + new
    done = self.spec.max_episode_steps <= self.simulation.count
    
    return observation,reward.detach().numpy(),done, {}

    
  def reset(self):
    if random.random()>.99:
        img= self.simulation.indexes.unsqueeze(-1)
        norm = (img-torch.min(img))/(torch.max(img) - torch.min(img))
        save(self.simulation.out(norm),str(random.random())+".png")
    self.simulation = PhysicSimulation(self.path,self.spp,self.frame_number,self.sppps,self.list,self.add,self.HEIGHT,self.WIDTH,self.max)
    return self.simulation.observe()
    
  def render(self, mode='human', close=False):
    return self.simulation.render()

def save(data,name):
    img= T.ToPILImage()(torch.Tensor(data).permute([2,0,1]))
    img.save(name)


import numpy as np
from typing import Dict, List
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()


class FCN(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):

        
        model_config["conv_filters"] = [
                                        [8,[7,7], [1,1]],
                                        [6,[7,7], [1,1]],
                                        [4,[7,7], [1,1]],
                                        [2,[7,7], [1,1]]]

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = "tanh" #self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        
        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        no_final_linear = True #self.model_config.get("no_final_linear")
        vf_share_layers = True

        self.last_layer_is_flattened = False
        self._logits = None

        layers = []
        (w, h, in_channels) = obs_space.shape

        in_size = [w, h]
        for out_channels, kernel, stride in filters:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

       # print(num_outputs)
        self._convs = nn.Sequential(*layers)

        
        self._value_branch = SlimFC(
                int(out_size[0]*out_size[1]*2), 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        s=conv_out.shape
        conv_out = conv_out.reshape(s[0], 1, -1).permute(0,2,1).squeeze(-1)
        self._features = conv_out 
    #    print(conv_out.shape)
        # Store features to save forward pass when getting value_function out.
 #       print(conv_out[0])
 #       print(conv_out[0].mean())
        return conv_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features.squeeze(-1)).squeeze(1)


    
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("FCN", FCN)



import ray

ray.init(num_gpus=4)

import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.ddppo as ddppo

import ray.rllib.algorithms.appo as appo
import ray.rllib.algorithms.a3c as a3c
import ray.rllib.algorithms.a2c as a2c
import ray.rllib.algorithms.ars as ars
import ray.rllib.algorithms.bc as bc
import ray.rllib.algorithms.crr as crr
import ray.rllib.algorithms.ddpg as ddpg
import ray.rllib.algorithms.es as es
import ray.rllib.algorithms.pg as pg
import ray.rllib.algorithms.mbmpo as mbmpo

from ray import serve
def train_ppo_model():
    a = time.time()
    algo = appo.APPO(env=CustomEnv,config={
'env_config':{'path': "../datasets/temple/",'number_images':None,\
'frame_number':1, 'spp':2, "sppps":.1
            },
          'framework' :"torch",
#"eager_tracing":True,

#"num_envs_per_worker":5,
        'num_workers':2,
"entropy_coeff":1e-3,
#"evaluation_num_workers":1,
#'num_cpus_per_worker':10,
'num_gpus_per_worker':2,
"evaluation_interval":5,
"rollout_fragment_length":20, #Increase this
"train_batch_size":40,
"replay_buffer_num_slots":50,

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


train_ppo_model()
