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
        c=21
        model_config["conv_filters"] = [
                                        [3,[c,c], [1,1]],
                                        [3,[c,c], [1,1]],
                                        [2,[c,c], [1,1]],
#                                        [2,[c,c], [1,1]],
#                                        [2,[c,c], [1,1]],
#                                        [2,[c,c], [1,1]],
                                        [1,[c,c], [1,1]],
                                        [2,[c,c], [1,1]]]

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = "tanh" #self.model_config.get("conv_activation")
        filters = model_config["conv_filters"]
        
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
        (in_channels,w, h) = obs_space.shape
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

        self._convs = nn.Sequential(*layers[:-2])
        self.head = nn.Sequential(*layers[-1:0])
        self.head_value = nn.Sequential(*layers[-2:-1])
        self._value_branch = SlimFC(
                int(out_size[0]*out_size[1]*2), 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        self._features = None
        GPUtil.showUtilization()

    def f(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        for i in range(int(input_dict["obs"].shape[0]/2)):
         x = input_dict["obs"][2*i:2*(i+1)]
         x = self._convs(x)
         tmp = self.head_value(x).reshape(*x.shape[:1],-1).mean(1)
         o=self.head(x)
         if i==0:
          self.tmp = tmp
          out=o
         else:
          out=torch.cat((out,o),0)
          self.tmp=torch.cat((self.tmp,tmp),0)
        return out


    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):


      if input_dict["obs"].shape[0]==8:
       with torch.no_grad():
          out=self.f(input_dict,state,seq_lens)
      else:
          out=self.f(input_dict,state,seq_lens)

      GPUtil.showUtilization()
      return out.reshape(input_dict["obs"].shape[0], -1), state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self.tmp is not None, "must call forward() first"
        return self.tmp


    
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("FCN", FCN)
