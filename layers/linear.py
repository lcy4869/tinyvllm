from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch


class LinearBase(nn.Module):
    def __init__(self, input_size, output_size, tp_dim: int | None = None):
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.input_size = input_size
        self.output_size = output_size
    def forward(self, x: torch.Tensor):
        raise NotImplementedError
    
class ColumnLinearBase(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, 0)
        self.output_size_parition = output_size // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.output_size_parition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if self.bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_parition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
    
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        loaded_weights = loaded_weights.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class MergedColumnLinear(ColumnLinearBase): #upgate_prj
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.input_size = input_size
        self.output_sizes = output_sizes
        super().__init__(self.input_size, sum(self.output_sizes), bias)
    
    def weight_loader(self, param, loaded_weights, loaded_shard_id: int):
        param_data = param.data
        offset = sum(self.output_sizes[:loaded_shard_id])//self.tp_size
        shard = self.output_sizes[loaded_shard_id]//self.tp_size
        param_data = param_data.narrow(self.tp_dim, offset, shard)
        loaded_weights = loaded_weights.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weights)
        

class QKVParallelLinear(ColumnLinearBase):
    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int,  total_num_kv_heads: int | None, bias: bool = False):
        tp_size = dist.get_world_size()
        self.head_size = head_size

        assert total_num_heads % tp_size == 0
        assert total_num_kv_heads % tp_size == 0
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        input_size = hidden_size
        output_size = self.num_heads*head_size + 2*self.num_kv_heads*head_size
        # self.weight = nn.Parameter(torch.empty(input_size, output_size))
        super().__init__(input_size, output_size, bias)
    
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, offset, shard_size)
        loaded_weights = loaded_weights.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weights)

class RowParallelLinear(LinearBase): # o proj, down proj
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, 1)
        assert input_size % self.tp_size == 0
        self.input_size_partition = input_size//self.tp_size
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        shard = param_data.size(self.tp_dim)
        offset = shard * self.tp_rank
        loaded_weights = loaded_weights.narrow(self.tp_dim, offset, shard)
        param_data.copy_(loaded_weights)
    
    def forward(self, x: torch.Tensor):
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            y = dist.all_reduce(y)
        return y
    



        








