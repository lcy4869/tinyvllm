from config import Config
from sampling_params import SamplingParams
from dataclasses import fields
from engine.scheduler import Scheduler
from engine.sequence import Sequence
from engine.model_runner import ModelRunner
from transformers import AutoTokenizer
import atexit
import torch.multiprocessing as mp


class LLM:
    # responsible for take request, encode, decode
    def __init__(self, model, **kwargs):
        field_names = {f.name for f in fields(Config)}
        field_values = {k: v for k, v in kwargs.items() if k in field_names}
        self.config = Config(model=model, **field_values)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.model = model
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, self.config.tensor_parallel_size):
            event = ctx.Event()
            p = ctx.Process(target=ModelRunner, args=(self.config, i, event))
            p.start()
            self.ps.append(p)
            self.events.append(event)
        self.model_runner = ModelRunner(self.config, 0, self.events)
        # model runner allocate kv cache so it is before scheduler init block manager.
        self.config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(self.config)
        atexit.register(self.exit_handler)

    def exit_handler(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()


    
    def add_requests(self, prompt, sampling_params: SamplingParams):
        # Convert string prompt to token IDs if needed
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = prompt
        s = Sequence(token_ids, sampling_params=sampling_params)
        self.scheduler.add(s)
        
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.post_process(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs)
        return outputs

    
    def generate(self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams]):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_requests(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0
        while not self.scheduler.is_finished():
            output = self.step()
            for seq_id, seq_output in output:
                outputs[seq_id] = seq_output
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids)} for token_ids in outputs]
        return outputs
