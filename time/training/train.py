import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model import *
from .dasr import get_model as dasr
from .ntas import get_model as ntas
from .model00 import get_model as a00
from .model01 import get_model as a01
from .model10 import get_model as a10
from .model21 import get_model as a21
from .model12 import get_model as a12


from .loss import *

# Worker function
def main_worker(inp=31, ic=32, mode=None, conf="111"): 
    """creates all objects needed to train the denoiser network

    Args:
        inp (int, optional): input number of channels. Defaults to 31.
        ic (int, optional): latent space number of channels. Defaults to 32.
        mode (str, optional): the variant/baseline being run. Defaults to None.
        conf (str, optional): the scale of every network. Defaults to "111". 0 means small, 1 default, 2 large.

    Returns:
        _type_: _description_
    """    
    if mode == "ntas":
        model = ntas(inp=inp, ic=ic)
    elif mode == "dasr":
        model = dasr(inp=inp, ic=ic)
    else:
        if conf[1:] == "01":
            model = a01( inp=inp, ic=ic)
        elif conf[1:] == "10":
            model = a10( inp=inp, ic=ic)
        elif conf[1:] == "21":
            model = a21( inp=inp, ic=ic)
        elif conf[1:] == "12":
            model = a12( inp=inp, ic=ic)
        elif conf[1:] == "00":
            model = a00( inp=inp, ic=ic)
        else:
            model = get_model( inp=inp, ic=ic)
    criterion = MixLoss([L1Loss(), MSSSIMLoss(weights=None)], [0.16, 0.84])

    optimizer = optim.Adam(model.parameters(), lr=1)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-05,
        total_steps=10 * 200 * 100,
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=(25.0),
        final_div_factor=1e4,
        last_epoch=-1,
    )
    valid_data = Dataset()
    return model, valid_data, criterion, optimizer, lr_scheduler
