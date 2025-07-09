import torch
from dataclasses import dataclass

@dataclass
class Context:
    is_prefill:bool = False
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, slot_mapping=None):
    _CONTEXT = Context(is_prefill=is_prefill, slot_mapping=slot_mapping)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()