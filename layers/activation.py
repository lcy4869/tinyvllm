from torch import nn
import torch.nn.functional as F
import torch

class SwiluAndMul(nn.Module):
    def __init__(self, config):
        super().__init__()
        
    @torch.compile
    def forward(self, x):
        x, y = x.chunk(2, -1)
        x = F.silu(x) * y
        return x
