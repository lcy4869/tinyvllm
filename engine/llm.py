from config import Config
from sampling_params import SamplingParams
from dataclasses import fields
from engine.scheduler import Scheduler
from engine.sequence import Sequence
from engine.model_runner import ModelRunner
from transformers import AutoTokenizer

class LLM:
    # responsible for take request, encode, decode
    def __init__(self, model, **kwargs):
        field_names = {f.name for f in fields(Config)}
        field_values = {k: v for k, v in kwargs.items() if k in field_names}
        self.config = Config(model=model, **field_values)
        self.model_runner = ModelRunner(self.config)
        self.scheduler = Scheduler(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.model = model

    
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
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.post_process(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs

    
    def generate(self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams]):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_requests(prompt, sp)
        outputs = {}
        while not self.scheduler.is_finished():
            output = self.step()
            for seq_id, seq_output in output:
                outputs[seq_id] = seq_output
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids)} for token_ids in outputs]
        return outputs
