
import os
import sys
from glob import glob
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from .config import *

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Trains a model using preprocessed datasets.')

  # Start the worker(s)
  start_workers(cfg, main_worker)

# Worker function
def main_worker():
  cfg = parse_args(description='Trains a model using preprocessed datasets.')
  model = get_model(cfg)
  model.to(device)
  criterion = get_loss_function(cfg)
  criterion.to(device)
  optimizer = optim.Adam(model.parameters(), lr=1)
  result_dir = get_result_dir(cfg)
  resume = os.path.isdir(result_dir)

  if resume:
    result_cfg = load_config(result_dir)
    last_epoch = get_latest_checkpoint_epoch(result_dir)
    checkpoint = load_checkpoint(result_dir, device, last_epoch, model, optimizer)
    step = checkpoint['step']

  # Initialize the validation dataset
  valid_data = ValidationDataset(cfg, cfg.valid_data)
  valid_steps_per_epoch = len(valid_data)
  progress_format = '%-5s %' + str(len(str(cfg.num_epochs))) + 'd/%d:' % cfg.num_epochs
  total_start_time = time.time()
  return model,valid_data
  for epoch in range(0,1):

    if True:
      # Validation
      if rank == 0:
        start_time = time.time()
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
