from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager
from collections import deque

class Scheduler:
    def __init__(self, config):
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_manager = BlockManager(config)

    def add(self, s: Sequence):
        self.waiting.append(s)
    
    def preempt(self, seq: Sequence):
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        seq.status = SequenceStatus.WAITING

    def schedule(self):
        #prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # check if it can scheduled
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            scheduled_seqs.append(seq)
            num_seqs += 1
            num_batched_tokens += len(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True
        
        #decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                scheduled_seqs.append(seq)
                self.block_manager.may_append(seq)
                self.running.appendleft(seq)
        assert scheduled_seqs
        return scheduled_seqs, False

    def post_process(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_tokens(token_id)
            if (not seq.ignore_eos and token_id == self.config.eos) or seq.num_completion_tokens >= self.config.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

            