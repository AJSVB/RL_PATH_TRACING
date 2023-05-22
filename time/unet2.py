import gym
from typing import Dict, List

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    same_padding,
    SlimConv2d,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import torchvision
from unet1 import *
torch, nn = try_import_torch()

class UN(nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        nn.Module.__init__(self)

        (temp, w, h) = obs_space.shape
        in_channels = 4
        self._convs = UNet(
            temp,in_channels,(
            in_channels,
            in_channels * 2,
            in_channels * 4,
            in_channels * 8,
            ),(
            in_channels * 8, 
            in_channels * 4, 
            in_channels * 2
            )
        ) 
        c = 3
        filters = [[1, [c, c], [1, 1]], [1, [c, c], [1, 1]]]
        layers = []
        in_size = [w, h]
        for out_channels, kernel, stride in filters:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    2,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn="linear", #relu
                )
            )
        self.head = nn.Sequential(*(layers[-2:1])) 

    def f(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        """Simple subfunction used in the forward call
        """  
        x = input_dict["obs"]
        mult = 1
        if (x == 0).all():
            mult = 0
        x = self._convs(x)
        o = self.head(x)
        return o * mult

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        """forward pass

        Args:
            input_dict (dict): contains the input data (observation)

        Returns:
            tensor: outputs the action recommendation (sampling heatmap)
        """    
        out = self.f(input_dict, state, seq_lens)
        out = out.reshape(out.shape[0], -1)
        return out, state
