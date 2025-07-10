import torch
from torch import nn
from transformers import Qwen3Config
from layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from layers.layernorm import RMSNorm
from layers.linear import MergedColumnLinear, QKVParallelLinear, RowParallelLinear
from layers.activation import SwiluAndMul
import torch.distributed as dist
from layers.attention import Attention
from layers.rotary_embedding import get_rope

class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, max_position: int = 4096 * 32):
            # def __init__(self, hidden_size: int, head_size: int, total_num_heads: int,  total_num_kv_heads: int | None, bias: bool = False):
        super().__init__()
        self.config = config
        self.qkv_proj = QKVParallelLinear(config.hidden_size, config.head_dim, config.num_attention_heads, config.num_key_value_heads, bias=False)
        self.q_norm = RMSNorm(config.head_dim)
        self.k_norm = RMSNorm(config.head_dim)
        self.rotary_emb = get_rope(config.head_dim,
            rotary_dim=config.head_dim,
            max_position=max_position,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.attn = Attention(config)
        self.o_proj = RowParallelLinear(config.head_dim*config.num_attention_heads, config.hidden_size, bias=False)
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = config.num_key_value_heads
        tp_size = dist.get_world_size()
        assert total_num_heads % tp_size == 0
        assert total_num_kv_heads % tp_size == 0
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        self.q_dim = self.num_heads * config.head_dim
        self.k_dim = self.num_kv_heads * config.head_dim
        self.head_dim = config.head_dim
    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_dim, self.k_dim, self.k_dim], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q_by_head)
        k = self.k_norm(k_by_head)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output
        

class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_up_proj = MergedColumnLinear(config.hidden_size, [config.intermediate_size]*2, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = SwiluAndMul(config)
    def forward(self, x: torch.Tensor):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)
    def forward(self, x: torch.tensor, positions: torch.Tensor, residual = None):
        if residual is None:
            residual = x
            x = self.input_layernorm(x)
        else:
            x, residual = self.input_layernorm(x, residual)
        x = self.self_attn(x, positions)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual

class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size)
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        x = self.embed_tokens(input_ids)
        residual = None
        for l in self.layers:
            x, residual = l(x, positions, residual)
        x, _ = self.norm(x, residual)
        return x

class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) :
        hidden_states =  self.model(input_ids, positions)
        return hidden_states
    
    def compute_logits(self, hidden_states: torch.Tensor):
        logits = self.lm_head(hidden_states)
        return logits

    