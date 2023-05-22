import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataset import *
from .util import *


def get_model(inp=31, ic=32):  # Was 31
    """adapter function between train.py and the model

    Args:
        inp (int, optional): the number of input channels. Defaults to 31.
        ic (int, optional): the number of latent space channels. Defaults to 32.

    Returns:
        Network: the described network
    """    
    return UNet(inp, ic=ic)


## -----------------------------------------------------------------------------
## U-Net model
## -----------------------------------------------------------------------------


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ic=32):
        """initialize the UNET as described

        Args:
            in_channels (int, optional): number of input channels. Defaults to 3.
            out_channels (int, optional): number of output channels. Defaults to 3.
            ic (int, optional): number of latent space channels. Defaults to 32.
        """        
        super(UNet, self).__init__()
        oc = out_channels

        self.enc_conv1a = Conv(in_channels + ic, 48)
        self.enc_conv1b = Conv(48, 48, 25)
        self.enc_conv2 = Conv(48, 48)
        self.enc_conv3 = Conv(48, 48)
        self.enc_conv4 = Conv(48, 48)
        self.enc_conv5 = Conv(48, 48)
        self.enc_conv6 = Conv(48, 96)
        self.dec_conv5a = Conv(96 + 48, 96)
        self.dec_conv5b = Conv(96, 96)
        self.dec_conv4a = Conv(96 + 48, 96)
        self.dec_conv4b = Conv(96, 96)
        self.dec_conv3a = Conv(96 + 48, 96)
        self.dec_conv3b = Conv(96, 96)
        self.dec_conv2a = Conv(96 + 48, 96)
        self.dec_conv2b = Conv(96, 64)
        self.dec_conv1a = Conv(64 + 48, 64, 0)
        self.dec_conv1b = Conv(64, oc, 0)

    def forward(self, input):
        x = relu(self.enc_conv1a(input))  
        x = relu(self.enc_conv1b(x))  
        x = pool1 = pool(x)  

        x = relu(self.enc_conv2(x))  
        x = pool2 = pool(x)  

        x = relu(self.enc_conv3(x))  
        x = pool3 = pool(x)  
        x = relu(self.enc_conv4(x))  
        x = pool4 = pool(x)  

        # Bottleneck
        x = relu(self.enc_conv5(x))  
        x = pool5 = pool(x)
        x = relu(self.enc_conv6(x))  
        x = pool(x)

        # Decoder
        # -------------------------------------------
        x = upsample(x)  
        x = concat(x, pool5)  
        x = relu(self.dec_conv5a(x))  
        x = relu(self.dec_conv5b(x))  

        x = upsample(x)  
        x = concat(x, pool4)  
        x = relu(self.dec_conv4a(x))  
        x = relu(self.dec_conv4b(x))  

        x = upsample(x)  
        x = concat(x, pool3)  
        x = relu(self.dec_conv3a(x))  
        x = relu(self.dec_conv3b(x))  

        x = upsample(x)  
        x = concat(x, pool2)  
        x = relu(self.dec_conv2a(x))  
        x = relu(self.dec_conv2b(x))  

        x = upsample(x)  
        x = concat(x, pool1)  
        x = relu(self.dec_conv1a(x))  
        x = relu(self.dec_conv1b(x))  
        x = upsample(x)[:, :, 20:-20, 20:-20]
        return x, input.squeeze(0)
