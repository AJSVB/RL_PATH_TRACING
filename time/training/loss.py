import torch
import torch.nn as nn
import torch.cuda.amp as amp

from .util import *
from pytorch_msssim import MS_SSIM

# L1 loss (seems to be faster than the built-in L1Loss)
class L1Loss(nn.Module):
    def forward(self, input, target):
        return torch.abs(input - target).mean()

# MS-SSIM loss
class MSSSIMLoss(nn.Module):
    def __init__(self, weights=None):
        super(MSSSIMLoss, self).__init__()
        self.msssim = MS_SSIM(data_range=1.0, weights=weights)

    def forward(self, input, target):
        with amp.autocast(enabled=False):
            return 1.0 - self.msssim(input.float(), target.float())

# Mix loss
class MixLoss(nn.Module):
    def __init__(self, losses, weights):
        super(MixLoss, self).__init__()
        self.losses = nn.Sequential(*losses)
        self.weights = weights

    def forward(self, input, target):
        return sum([l(input, target) * w for l, w in zip(self.losses, self.weights)])
