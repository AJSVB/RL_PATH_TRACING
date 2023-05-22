import torch
import torch.nn as nn
import torch.nn.functional as F
import time
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
        add_networks(self, in_channels, ic, out_channels, .7)

    def forward(self, input):
        a = time.time()
        # Backbone
        # ---------------------
        input = relu(self.a(input))
        #    input = relu(self.a1(input))
        #    input = relu(self.b(input))
        #    input = relu(self.b1(input))
        input = self.c(input)
        input = tanh(input)
        torch.cuda.synchronize()
        print("state " + str(time.time() - a))
        a = time.time()
        x=forward(self, input)
        torch.cuda.synchronize()
        print("decoder " + str(time.time() - a))
        return x, input.squeeze(0)
