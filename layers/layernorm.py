import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, 
                 hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var+self.eps)
        x = x.to(orig_dtype) * self.weight
        return x
    
    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor):
        orig_dtype = x.dtype
        x = x.to(torch.float32) + residual.to(torch.float32)
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        x = x * torch.rsqrt(var)
        x = x.to(orig_dtype) * self.weight
        return x, residual
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


        