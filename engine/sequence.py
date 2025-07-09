from sampling_params import SamplingParams
from copy import copy
from enum import Enum, auto
from itertools import count

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    counter = count()
    def __init__(self, token_ids: list[int], block_size=256, sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.blocks_table = []  # store block id
        self.tokens_ids = copy(token_ids)
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.max_tokens = sampling_params.max_tokens
        self.temperature = sampling_params.temperature
        self.ignore_eos = sampling_params.ignore_eos
        self.status = SequenceStatus.WAITING
        self.block_size = block_size
        
    def __getitem__(self, key):
        return self.tokens_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED
    
    @property
    def last_token(self):
        return self.tokens_ids[-1]
    
    def append_tokens(self, token_id):
        self.tokens_ids.append(token_id)
        self.num_tokens += 1
    
    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size
    
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    @property
    def num_completion_tokens(self):
        return self.num_tokens - len(self.num_prompt_tokens)

    
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.tokens_ids[i*self.block_size: (i+1)*self.block_size]

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks-1) * self.block_size 
