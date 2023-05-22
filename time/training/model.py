# Copyright 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

# -----------------------------------------------------------------------------
# U-Net model
# -----------------------------------------------------------------------------


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ic=32):
        """initialize the UNET as described

        Args:
            in_channels (int, optional): number of input channels. Defaults to 3.
            out_channels (int, optional): number of output channels. Defaults to 3.
            ic (int, optional): number of latent space channels. Defaults to 32.
        """      
        print(in_channels)
        super(UNet, self).__init__()
        d = 1
        # Number of channels per layer
        ic = ic
        ec1 = 32
        ec2 = 48 // d
        ec3 = 64 // d
        ec4 = 80 // d
        ec5 = 96 // d
        dc4 = 112 // d
        dc3 = 96 // d
        dc2 = 64 // d
        dc1a = 64 // d
        dc1b = 32 // d
        oc = out_channels

        # Convolutions
        inc = in_channels + ic

        nb = 64
        if ic == 0:
            ic = ec1

        self.enc_conv0 = Conv(ic, ec1)
        #    self.enc_conv0  = Conv(64,      ec1)
        self.enc_conv1 = Conv(ec1, ec1)
        self.enc_conv2 = Conv(ec1, ec2)
        self.enc_conv3 = Conv(ec2, ec3)
        self.enc_conv4 = Conv(ec3, ec4)
        self.enc_conv5a = Conv(ec4, ec5)
        self.enc_conv5b = Conv(ec5, ec5)
        self.dec_conv4a = Conv(ec5 + ec3, dc4)
        self.dec_conv4b = Conv(dc4, dc4)
        self.dec_conv3a = Conv(dc4 + ec2, dc3)
        self.dec_conv3b = Conv(dc3, dc3)
        self.dec_conv2a = Conv(dc3 + ec1, dc2)
        self.dec_conv2b = Conv(dc2, dc2)
        self.dec_conv1a = Conv(dc2 + ic, dc1a)
        self.dec_conv1b = Conv(dc1a, dc1b)
        self.dec_conv0 = Conv(dc1b, oc)

        self.a = SimpleConv(inc, nb)
        self.a1 = Conv(nb, nb)
        self.b = Conv(nb, nb)
        self.b1 = Conv(nb, nb)
        self.c = SimpleConv(nb, ic)

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
#        print("state " + str(time.time() - a))
        a = time.time()
        # Encoder
        # -------------------------------------------
        x = relu(self.enc_conv0(input))  # enc_conv0

        x = relu(self.enc_conv1(x))  # enc_conv1
        x = pool1 = pool(x)  # pool1

        x = relu(self.enc_conv2(x))  # enc_conv2
        x = pool2 = pool(x)  # pool2

        x = relu(self.enc_conv3(x))  # enc_conv3
        x = pool3 = pool(x)  # pool3

        x = relu(self.enc_conv4(x))  # enc_conv4
        x = pool(x)  # pool4

        # Bottleneck
        x = relu(self.enc_conv5a(x))  # enc_conv5a
        x = relu(self.enc_conv5b(x))  # enc_conv5b

        # Decoder
        # -------------------------------------------

        x = upsample(x)  # upsample4
        x = concat(x, pool3)  # concat4
        x = relu(self.dec_conv4a(x))  # dec_conv4a
        x = relu(self.dec_conv4b(x))  # dec_conv4b

        x = upsample(x)  # upsample3
        x = concat(x, pool2)  # concat3
        x = relu(self.dec_conv3a(x))  # dec_conv3a
        x = relu(self.dec_conv3b(x))  # dec_conv3b

        x = upsample(x)  # upsample2
        x = concat(x, pool1)  # concat2
        x = relu(self.dec_conv2a(x))  # dec_conv2a
        x = relu(self.dec_conv2b(x))  # dec_conv2b

        x = upsample(x)  # upsample1
        x = concat(x, input)  # concat1
        x = relu(self.dec_conv1a(x))  # dec_conv1a
        x = relu(self.dec_conv1b(x))  # dec_conv1b

        x = self.dec_conv0(x)  # dec_conv0
        torch.cuda.synchronize()
#        print("decoder " + str(time.time() - a))
        return x, input.squeeze(0)
