from engine.sequence import Sequence
import xxhash
import numpy as np
from collections import deque

class Block:
    def __init__(self, block_id):
        self.id = block_id
        self.ref_count = 0
        self.token_ids = []
        self.hash = -1

    def update(self, hash:int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

class BlockManager:
    def __init__(self, config):
        self.free_blocks: deque[int] = deque(range(config.num_kvcache_blocks))
        self.used_blocks: set[int] = set()
        self.num_blocks = config.num_kvcache_blocks
        self.block_size = config.kvcache_block_size
        self.blocks = [Block(i) for i in range(self.num_blocks)]
        self.hash_to_block_id = {}

    def compute_hash(cls, token_ids: list[int], prefix_hash: int=-1):
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
    
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_blocks.remove(block_id)
        self.used_blocks.add(block_id)
        return block

    def can_allocate(self, seq: Sequence):
        # it does not consider prefix cache
        print(f"can_allocate: {len(self.free_blocks)} >= {seq.num_blocks}")
        return len(self.free_blocks) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_tables
        cache_miss = False
        hash = -1 # hash
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            hash = self.compute_hash(token_ids, hash) if len(token_ids) == self.block_size else -1
            block_id =  self.hash_to_block_id.get(hash, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_blocks[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            seq.block_tables.append(block_id)
            if hash != -1:
                block.update(hash, token_ids)
                self.hash_to_block_id[hash] = block_id

    def _deallocate(self, block_id):
        assert self.blocks[block_id].ref_count == 0
        self.free_blocks.append(block_id)
        self.used_blocks.remove(block_id)

    def deallocate(self, seq: Sequence):
        for i in range(seq.num_blocks):
            block_id = seq.num_blocks[i]
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate(block_id)
        seq.num_cached_tokens = 0
        seq.blocks_table.clear()
    
    def can_append(self, seq: Sequence):
        len(self.free_blocks) >= (len(seq) % self.block_size == 1)
    
    def may_append(self, seq: Sequence):
        # append to last block, or create a new one and append last token
        block_tables = seq.blocks_table
        last_block = self.blocks[block_tables[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_blocks[0]
            self._allocate_block(block_id)
            seq.blocks_table.append(block_id)
            # not full block, so do not update hash
        elif len(seq) % self.block_size == 0:
            # update last block hash
            assert last_block.hash == -1
            prefix = self.blocks[block_tables[-2]] if len(block_tables) >  1 else -1
            token_ids = seq.block(seq.num_blocks-1)
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1








            
