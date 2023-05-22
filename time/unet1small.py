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
torch, nn = try_import_torch()
from unet1 import *
class UN(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        """Adapter class for UNET wrt rayRLlib

        Args:
            obs_space (gym.spaces.Space): the dimensions of the observation space
            action_space (gym.spaces.Space): the dimensions of the action space
            num_outputs (int): the dimension of the output (1D)
            model_config (ModelConfigDict): Description of the model's configuration
            name (str): name of the model
        """    
        if model_config is not None:
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
        nn.Module.__init__(self)

        (temp, w, h) = obs_space.shape
        in_channels=1
        self._convs = UNet(
            temp,in_channels,
             [in_channels, in_channels * 2],
         [in_channels * 2]
        ) 
        c = 3
        filters = [[2, [c, c], [1, 1]], [1, [c, c], [1, 1]]]
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
                    activation_fn="relu",
                )
            )

        # layers.append(nn.Tanh())
        # in_channels = out_channels
        self.head = nn.Sequential(*(layers[-2:1]))
        self.head_value = nn.Sequential(*layers[-1:0])

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
        self.tmp = self.head_value(x).reshape(*x.shape[:1], -1).mean(1)
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
        temp = torch.nn.Tanh()(out) * 20 - 10
        return temp.to(torch.float), state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """value function

        Returns:
            TensorType: the value function
        """        
        assert self.tmp is not None, "must call forward() first"
        return self.tmp.to(torch.float)


from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("UNsmall", UN)
