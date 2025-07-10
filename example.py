import os
from engine.llm import LLM
from sampling_params import SamplingParams
from transformers import AutoTokenizer
import time

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    total_tokens = sum(len(seq) for seq in outputs)
    throughput = total_tokens / (end_time - start_time)
    print(f"Throughput: {throughput} tokens/s")

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
