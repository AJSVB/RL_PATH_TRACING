import time
from typing import List

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

class CNN(nn.Module):
    def __init__(self):
        """This CNN with filter size of 1 acts as a FCN to every individual pixels
        """        
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(42, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = self.leakyrelu(x)
        return x

class FCN(nn.Module):
    def __init__(self):
        """Acts as described by IMCD.
        Given an input with additional data, previous embedding, and new data, 
        computes new embedding by passing every input data frame
        using a FCN, then averages the result.
        """
        super(FCN, self).__init__()
        self.CNN = CNN()

    def forward(
        self,
        x: TensorType,
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        if (x == 0).all():
            pass
        x_add = x[
            :39
        ]  # 32 channels for previous embedding, 7 channels for additional data
        x = [
            self.CNN(torch.cat((x_add, x[39 + 3 * i : 42 + 3 * i]), 0).unsqueeze(0))
            for i in range(8)
        ]
        x = torch.cat(x, 0)
        out = torch.mean(x, 0)
        return out, state
