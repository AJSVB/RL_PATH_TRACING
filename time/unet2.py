
import gym
from typing import Dict, List

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
import torchvision
import GPUtil

torch, nn = try_import_torch()


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3,padding=1)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(6,8,16,32,64,128)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(128,64, 32, 16, 8)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2,padding=0,output_padding=0) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, in_channels, enc_chs=(6,8,8,16,32), dec_chs=(64, 32, 16, 8), num_class=2, retain_dim=False, out_sz=(720,1280)):
        super().__init__()
        temp=in_channels
        in_channels=6
        enc_chs=(in_channels,in_channels*2,in_channels*4,in_channels*8,in_channels*16)
        dec_chs=(in_channels*16,in_channels*8,in_channels*4,in_channels*2)

        self.tail        = nn.Conv2d(temp, in_channels, 3,padding=1)
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 3,padding=1)

    def forward(self, x):
        x = self.tail(x)
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        return out
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

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

        
        (in_channels,w, h) = obs_space.shape
        self._convs = UNet(in_channels) #DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=2, planes=32, spp_planes=128, head_planes=64, augment=in_channels)




        c=3
        filters = [                     [1,[c,c], [1,1]],
                                        [1,[c,c], [1,1]]]
        layers=[]
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
         x = input_dict["obs"]
         mult = 1
         if (x==0).all():
           mult = 0
         x = self._convs(x)
         self.tmp = self.head_value(x).reshape(*x.shape[:1],-1).mean(1)
         o=self.head(x)
         return o*mult

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
      import time

      time.time()
      out=self.f(input_dict,state,seq_lens)
      out=out.reshape(out.shape[0],-1)
      return out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self.tmp is not None, "must call forward() first"
        return self.tmp.to(torch.float)


