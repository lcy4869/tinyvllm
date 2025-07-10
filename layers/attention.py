from torch import nn
import torch
from utils.context import get_context
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache, flash_attn_func

@triton.jit
def store_kvcache_kernel(k_ptr, k_stride, v_ptr, v_stride, k_cache, v_cache, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)
    key_offset = idx * k_stride + tl.arange(0, D)
    k = tl.load(k_ptr + key_offset)
    value_offset = idx * v_stride + tl.arange(0, D)
    v = tl.load(v_ptr + value_offset)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offset = slot * D + tl.arange(0, D)
    tl.store(k_cache + cache_offset, k)
    tl.store(v_cache + cache_offset, v)

def store_kv(k: torch.Tensor, v: torch.Tensor,  k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = k.shape
    D = num_heads * head_dim
    assert k.stride(-1) == 1 and v.stride(-1) == 1
    assert k.stride(1) == head_dim and v.stride(1) == head_dim
    # len, head, dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](k, k.stride(0), v, v.stride(0), k_cache, v_cache, slot_mapping, D)

    
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k_cache = self.v_cache = torch.tensor([])
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

    def forward(self, q, k, v):
        context = get_context()
        ori_dtype = q.dtype
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        store_kv(k, v, self.k_cache, self.v_cache, context.slot_mapping)
        block_tables = context.block_tables
        k_cache, v_cache = self.k_cache, self.v_cache
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            
            o = flash_attn_varlen_func(q, k, v, 
                                       cu_seqlens_q=context.cu_seqlens_q, cu_seqlens_k=context.cu_seqlens_k, 
                                       max_seqlen_q=context.max_seqlen_q, max_seqlen_k=context.max_seqlen_k,
                                       causal=True, block_table=context.block_tables)
        else:
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens=context.context_lens, block_table=context.block_tables)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o

