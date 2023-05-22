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
from ray.rllib.models import ModelCatalog
torch, nn = try_import_torch()

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Componenent of the UNET that contains two Conv layers with a ReLU layer between them.

        Args:
            in_ch (int): number of in channels
            out_ch (int): number of out channels
        """        
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(nn.Module):
    def __init__(self, chs=(6, 8, 16, 32, 64, 128)):
        """the encoding component of the UNET (compresses)

        Args:
            chs (tuple, optional): the latent number of channels. Defaults to (6, 8, 16, 32, 64, 128).
        """        
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(128, 64, 32, 16, 8)):
        """the decoding component of the UNET (expands)

        Args:
            chs (tuple, optional): the latent number of channels. Defaults to (128, 64, 32, 16, 8).
        """        
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    chs[i], chs[i + 1], 2, 2, padding=0, output_padding=0
                )
                for i in range(len(chs) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(
        self,
        temp,
        in_channels,
        enc_chs,
        dec_chs,
        ):
        """The UNet: initial conv layer, compressing part, expanding part, and final conv layer

        Args:
            temp (int): the intitial number of channels of the input data
            in_channels (int): the input number of channels for the unet
            enc_chs (list): list of int describing the number of latent channels in the encoding part
            dec_chs (list): list of int describing the number of latent channels in the decoding part
        """        
        super().__init__()
        self.tail = nn.Conv2d(temp, in_channels, 3, padding=1)
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], 2, 3, padding=1)

    def forward(self, x):
        x = self.tail(x)
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        return out

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
        (in_channels, in_channels * 2, in_channels * 4),
        (in_channels * 4, in_channels * 2)
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
        temp = torch.nn.Tanh()(out) * 20 - 10  # We want bounded outputs (see paper)
        return temp.to(torch.float), state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """value function

        Returns:
            TensorType: the value function
        """        
        assert self.tmp is not None, "must call forward() first"
        return self.tmp.to(torch.float)

ModelCatalog.register_custom_model("UN", UN)
