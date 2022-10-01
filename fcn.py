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
from torch.nn import Conv2d
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
        filters = [
                                        [2,[c,c], [1,1]],
   #                                     [4,[c,c], [1,1]],
   #                                     [4,[c,c], [1,1]],
   #                                     [4,[c,c], [1,1]],
          #                              [2,[c,c], [1,1]],
                                        #[5,[c,c], [1,1]],
                                        [2,[c,c], [1,1]],
#                                        [2,[c,c], [1,1]],
#                                        [2,[c,c], [1,1]],
                                        [2,[c,c], [1,1]],
                                        [2,[c,c], [1,1]]]

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = "tanh" 

        layers = []
        (in_channels,w,h) = obs_space.shape
 #       print("nothing happend")
 #       GPUtil.showUtilization()
        in_size = [w, h]
        for out_channels, kernel, stride in filters:
            padding, out_size = same_padding(in_size, kernel, stride)
            print(in_size)
            print(kernel)
            print(stride)
            print(padding)
            print(out_size)
            layers.append(
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding[:2], #Super strange
                    #activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size
            layers.append(get_activation_fn(activation, "torch")()) 

        #GPUtil.showUtilization()
        self._convs = nn.Sequential(*layers[:4])
        self.head = nn.Sequential(*layers[-2:0])
        self.head_value = nn.Sequential(*layers[-4:-2])
        #GPUtil.showUtilization()
#        self._value_branch = SlimFC(
#                int(out_size[0]*out_size[1]*2), 1, initializer=normc_initializer(0.01), activation_fn=None
#            )
        self._features = None
        GPUtil.showUtilization()

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
#        print("forward")
#        GPUtil.showUtilization()
#        print(input_dict["obs"].float().shape)
        x = input_dict["obs"]
#        GPUtil.showUtilization()
        x = self._convs(x)
#        GPUtil.showUtilization()
        self._features = self.head_value(x).reshape(x.shape[0], 1, -1).mean(-1).squeeze(-1)
#        GPUtil.showUtilization()
        x=self.head(x) 
#        print("end forward")
#        GPUtil.showUtilization()
        return x.reshape(x.shape[0], 1, -1).permute(0,2,1).squeeze(-1), state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._features 


    
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("FCN", FCN)
