
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
        c=1
        model_config["conv_filters"] = [
#                                        [64,[c,c], [1,1]],
                                        [3,[c,c], [1,1]],
#                                        [64,[c,c], [1,1]],
#                                        [64,[c,c], [1,1]]
]

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
#            layers.append(nn.ReLU())
            in_channels = out_channels
            in_size = out_size
        self.num_outputs = w*h
        self._convs = nn.Sequential(*layers) #.cuda(0)
        #GPUtil.showUtilization()
    def f(
        self,
        x,
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
         out = self._convs(x)
         return out


    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
      x=input_dict["obs"].type(torch.float32)
      x=x.type(torch.float32)
      out=self.f(x,state,seq_lens)
      return  out, state

    
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("FCN1", FCN)
