
import GPUtil
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import time
import torchvision.transforms as T
import random
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

from torch import nn 

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
        c=3
        model_config["conv_filters"] = [
                                        [20,[c,c], [1,1]],
                                        [18,[c,c], [1,1]],
                                        [16,[c,c], [1,1]],
                                        [14,[c,c], [1,1]],
                                        [12,[c,c], [1,1]],
                                        [10,[c,c], [1,1]],
                                        [8,[c,c], [1,1]],
                                        [6,[c,c], [1,1]],
                                        [4,[c,c], [1,1]],
                                        [1,[c,c], [1,1]],
                                        [1,[c,c], [1,1]],
                                        [1,[c,c], [1,1]]]

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        filters = model_config["conv_filters"]

        layers = []
        (in_channels,w, h) = obs_space.shape
        in_size = [w, h]
        for out_channels, kernel, stride in filters:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding[:2],
                #    activation_fn=activation,
                )
            )
            layers.append(nn.Tanh())
            in_channels = out_channels
            in_size = out_size

        self._convs = nn.Sequential(*layers[:-4])#.cuda(0)
        self.head = nn.Sequential(*layers[-2:0])#.cuda(0)
        self.head_value = nn.Sequential(*layers[-4:-2])#.cuda(0)
        #GPUtil.showUtilization()
        self.CST=2
    def f(
        self,
        y,
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
#        CST=min(self.CST,y.shape[0])
#        for i in range(int(y.shape[0]/CST)):
         x = y #[CST*i:CST*(i+1),]
         x = self._convs(x)
         tmp = self.head_value(x).reshape(*x.shape[:1],-1).mean(1)
         o=self.head(x)
        # if i==0:
         self.tmp = tmp
         out=o
        # else:
        #  out=torch.cat((out,o),0)
        #  self.tmp=torch.cat((self.tmp,tmp),0)
         return out


    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
      x=input_dict["obs"]
#      print(x.shape)
      out=self.f(x,state,seq_lens)
      t = out *  0 -10  
      out=torch.cat((out,t),1)
      out = out.reshape(input_dict["obs"].shape[0], -1)
  #    print(torch.isnan(out).any())
      return  out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self.tmp is not None, "must call forward() first"
 #       print(torch.isnan(self.tmp).any())
        return self.tmp


    
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("FCN", FCN)
