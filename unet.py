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
            x = self.pool(x)
            ftrs.append(x)
        return ftrs
class Decoder(nn.Module):
    def __init__(self, chs=(128,64, 32, 16, 8)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2,padding=0,output_padding=0) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-2):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        x=self.upconvs[len(self.chs)-2](x)
        return x
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
class UNet(nn.Module):
    def __init__(self, in_channels, enc_chs=(6,8,8,16,32), dec_chs=(64, 32, 16, 8), num_class=2,  retain_dim=False, out_sz=(720,1280)):
        super().__init__()
        enc_chs=(2,4) #,8) #,in_channels*8)
        dec_chs =(4,2)
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(enc_chs[0], num_class, 3,padding=1)
        self.pre =                 SlimConv2d(
                    in_channels,
                    2,
                    3,
                    1,
                    1,
                    activation_fn="linear",
                )


    def forward(self, x):
        x=self.pre(x)
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
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


        (w, h, in_channels) = obs_space.shape
        self._convs = UNet(in_channels) #DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=2, planes=32, spp_planes=128, head_planes=64, augment=in_channels)

        
        self._value_branch = SlimFC(
                int(w*h*2), 1, initializer=normc_initializer(0.01), activation_fn=None
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
ModelCatalog.register_custom_model("UN", UN)
