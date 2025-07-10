import torch
from dataclasses import dataclass

@dataclass
class Context:
    is_prefill:bool = False
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, slot_mapping=None, block_tables=None, context_lens=None, cu_seqlens_k=None, cu_seqlens_q=None, max_seqlen_q=None, max_seqlen_k=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill=is_prefill, slot_mapping=slot_mapping, block_tables=block_tables, context_lens=context_lens, cu_seqlens_k=cu_seqlens_k, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()