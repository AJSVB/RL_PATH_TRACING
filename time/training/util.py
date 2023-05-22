import torch
import torch.nn as nn
import torch.nn.functional as F
## -----------------------------------------------------------------------------
## Network layers
## -----------------------------------------------------------------------------


# 3x3 convolution module
def Conv(in_channels, out_channels, padding=1):
    return nn.Conv2d(in_channels, out_channels, 3, padding=padding)


# 1x1 convolution module
def SimpleConv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 1, padding=0)


# ReLU function
def relu(x):
    return F.relu(x, inplace=True)

# Tanh activation function
def tanh(x):
    return torch.tanh(x)

# 2x2 max pool function
def pool(x):
    return F.max_pool2d(x, 2, 2)


# 2x2 nearest-neighbor upsample function
def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


# Channel concatenation function
def concat(a, b):
    return torch.cat((a, b), 1)


def add_networks(self,in_channels,ic,out_channels,d):
    ic = ic
    ec1 = 32
    ec2 = int(48 // d)
    ec3 = int(64 // d)
    ec4 = int(80 // d)
    ec5 = int(96 // d)
    dc4 = int(112 // d)
    dc3 = int(96 // d)
    dc2 = int(64 // d)
    dc1a = int(64 // d)
    dc1b = int(32 // d)
    oc = out_channels
    nb = 64
    if ic == 0:
        ic = ec1
    # Convolutions
    inc = in_channels + ic
    self.enc_conv0 = Conv(ic, ec1)
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
    self.dir = SimpleConv(inc, ic)


def forward(self,input):
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
    return x