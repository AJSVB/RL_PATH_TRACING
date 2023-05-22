import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from .dataset import *
from .util import *

def get_model(inp=31, ic=32):
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
        # Number of channels per layer
        oc = out_channels

        self.enc_conv1a = Conv(10, 32)
        self.enc_conv1b = Conv(32, 32)
        self.enc_conv2a = Conv(32, 43)
        self.enc_conv2b = Conv(43, 43)
        self.enc_conv3a = Conv(43, 57)
        self.enc_conv3b = Conv(57, 57)
        self.enc_conv4a = Conv(57, 76)
        self.enc_conv4b = Conv(76, 76)
        self.enc_conv5a = Conv(76, 76)
        self.enc_conv5b = Conv(76, 76)

        self.dec_conv4a = Conv(76 * 2, 57)
        self.dec_conv4b = Conv(57, 57)
        self.dec_conv3a = Conv(57 * 2, 43)
        self.dec_conv3b = Conv(43, 43)
        self.dec_conv2a = Conv(43 * 2, 32)
        self.dec_conv2b = Conv(32, 32)
        self.dec_conv1a = Conv(32 * 2, 64)
        self.dec_conv1b = Conv(64, 64)
        self.dec_conv0 = Conv(64, oc)

    def forward(self, input):
        x = relu(self.enc_conv1a(input))  
        x = relu(self.enc_conv1b(x))  
        x, pool1 = pool(x), x  
        x = relu(self.enc_conv2a(x))  
        x = relu(self.enc_conv2b(x))  
        x, pool2 = pool(x), x
        x = relu(self.enc_conv3a(x))  
        x = relu(self.enc_conv3b(x))  
        x, pool3 = pool(x), x
        x = relu(self.enc_conv4a(x))  
        x = relu(self.enc_conv4b(x))  
        x, pool4 = pool(x), x
        x = relu(self.enc_conv5a(x))  
        x = relu(self.enc_conv5b(x))  

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
        x = self.dec_conv0(x)  
        return x, input.squeeze(0)
