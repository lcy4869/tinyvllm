from torch import nn
import torch.nn.functional as F

class SwiluAndMul(nn.Module):
    def __init__(self, config):
        self.config = config
        self.hidden_size = config.hidden_size
    def forward(self, x):
        x, y = x.chunk(2, -1)
        x = F.silu(x) * y
        return x
