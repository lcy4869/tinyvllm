from engine.sequence import Sequence
from layers.sampler import Sampler
import torch
from utils.context import set_context, get_context, reset_context
from utils.loader import load_model
from models.qwen import Qwen3ForCausalLM
import torch.distributed as dist


class ModelRunner:
    def __init__(self, config, rank: int = 0):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.kv_cache = None
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(self.config.hf_config)
        self.sampler = Sampler()
        self._allocate_kv_cache()
        self.enforce_eager = config.enforce_eager
        print(config.enforce_eager)
        if not config.enforce_eager:
            print("capture cudagraph")
            self.capture_cudagraph()
            
        load_model(self.model, config.model)
    
    def _allocate_kv_cache(self):
        # get gpu memory
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        torch_allocated = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        non_torch_allocated = used - torch_allocated
        peak_memory += non_torch_allocated
        available_kv_cache_memory = total*config.gpu_memory_utilization - peak_memory
        num_nv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_nv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(available_kv_cache_memory) // block_bytes 
        assert config.num_kvcache_blocks > 0

        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_nv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs):
        max_len = max(len(seq.block_tables) for seq in seqs)
        block_tables = [seq.block_tables + [-1] * (max_len-len(seq.block_tables)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs):
        # concat into batch*seqlen, token_id
        input_ids = []
        positions_ids = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        block_tables = None
        slot_mapping = []
        max_seqlen_q = 0
        max_seqlen_k = 0
        for seq in seqs:
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions_ids.extend(list(range(seq.num_cached_tokens, len(seq))))
            seqlens_q = len(seq) - seq.num_cached_tokens
            seqlens_k = len(seq)
            max_seqlen_q = max(max_seqlen_q, seqlens_q)
            max_seqlen_k = max(max_seqlen_k, seqlens_k)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k)
            if not seq.block_tables:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start_idx = seq.block_tables[i]*self.block_size
                if i != seq.num_blocks -1:
                    end_idx = start_idx + self.block_size
                else:
                    end_idx = start_idx + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start_idx, end_idx)))
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_ids = torch.tensor(positions_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(is_prefill=True, cu_seqlens_k=cu_seqlens_k, cu_seqlens_q=cu_seqlens_q, block_tables=block_tables, slot_mapping=slot_mapping, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k)
        return input_ids, positions_ids

    def prepare_decode(self, seqs):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_tables[-1]*self.block_size + seq.last_block_num_tokens-1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(is_prefill=False, context_lens=context_lens, slot_mapping=slot_mapping, block_tables=block_tables)
        return input_ids, positions
    
    def run_model(self, input_ids, postion_ids, is_prefill):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, postion_ids))
        else:
            bs = input_ids.size(0)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            context = get_context()
            for k, v in graph_vars.items():
                if k !="output":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["positions"][:bs] = postion_ids
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(self.graph_vars["outputs"][:bs])

             
    def prepare_sample(self, seqs):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures
    
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, postion_ids = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # bs, len=1, vocab_size
        logits = self.run_model(input_ids, postion_ids, is_prefill)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank==0 else None
        return token_ids
    
    @torch.inference_mode()
    def capture_cudagraph(self): # for decoding
        max_bs = min(self.config.max_num_seqs, 512)
        hf_config = self.config.hf_config
        max_num_blocks = (self.config.max_model_len + self.block_size - 1)//self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        content_lens = torch.zeros(max_bs, dtype=torch.int32)
        self.graph_pool = None
        self.graphs = {}
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs+1, 16))


        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], block_tables=block_tables[:bs], context_lens=content_lens[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs]) # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            outputs=outputs,
            block_tables=block_tables,
            content_lens=content_lens
        )
