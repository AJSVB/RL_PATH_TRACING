import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.distributed as dist
#from torch.utils.tensorboard import SummaryWriter

from .config import *
from .model import *
from .result import *
from .loss import *

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Trains a model using preprocessed datasets.')

  # Start the worker(s)
  start_workers(cfg, main_worker)

# Worker function
def main_worker(inp = 33,ic=32):
  cfg = parse_args(description='Trains a model using preprocessed datasets.')
  model = get_model(cfg,inp=inp,ic=ic)
  criterion = get_loss_function(cfg)


  optimizer = optim.Adam(model.parameters(), lr=1)
  result_dir = get_result_dir(cfg)
  os.path.isdir(result_dir)

  def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
  model.apply(init_weights)

  lr_scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=cfg.max_lr,
    total_steps=cfg.num_epochs*200*100,
    pct_start=cfg.lr_warmup,
    anneal_strategy='cos',
    div_factor=(25. if cfg.lr is None else cfg.max_lr / cfg.lr),
    final_div_factor=1e4,
    last_epoch=-1)

  # Initialize the validation dataset
  valid_data = ValidationDataset(cfg, cfg.valid_data)
  valid_steps_per_epoch = len(valid_data)
  progress_format = '%-5s %' + str(len(str(cfg.num_epochs))) + 'd/%d:' % cfg.num_epochs
  time.time()
  return model,valid_data,criterion,optimizer,lr_scheduler
  for epoch in range(0,1):

    if True:
      # Validation
      if rank == 0:
        time.time()
        progress = ProgressBar(valid_steps_per_epoch, progress_format % ('Valid', epoch))

      # Switch to evaluation mode
      model.eval()
      valid_loss = 0.

      # Iterate over the batches
      with torch.no_grad():
        for i in range(len(valid_data)):
          batch = valid_data[i]
          # Get the batch
          input, target = batch
          input  = input.to(device,  non_blocking=True).float().unsqueeze(0)
          target = target.to(device, non_blocking=True).float().unsqueeze(0)

          # Run a validation step
          loss = criterion(model(input), target)

          # Next step
          valid_loss += loss
          if rank == 0:
            progress.next()

  # Cleanup
  cleanup_worker(cfg)

if __name__ == '__main__':
  main()





