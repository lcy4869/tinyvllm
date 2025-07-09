from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class Config:
    model: str
    max_model_len: int = 4096
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 16384 # gpu can store
    tensor_parallel_size: int = 1
    hf_config: AutoConfig | None = None 
    kvcache_block_size: int = 256
    gpu_memory_utilization: float = 0.9
    num_kvcache_blocks: int = -1
    enforce_eager: bool = False
    eos: int = -1


