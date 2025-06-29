import argparse
import numpy as np
import os
import torch
import json
from collections import Counter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import re

def apply_chat_template(toker, messages):
    input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return toker(input_prompt, add_special_tokens=False).input_ids

def prepare_input_boxed(template, input_d):
    problem = input_d['problem']
    steps = input_d['steps']
    tagged_response = ''
    for sdx, step in enumerate(steps):
        tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
    tagged_response = tagged_response.strip()
    prompt = template.format(problem=problem, tagged_response=tagged_response)
    messages = [{'role': 'user', 'content': prompt}]
    return messages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        choices=['gsm8k', 'math', 'olympiadbench', 'omnimath'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='./outputs')
    parser.add_argument('--use_voting', action='store_true')
    parser.add_argument('--voting_n', type=int, default=8)
    parser.add_argument('--tensor_parallel_size', type=int, default=4)
    args = parser.parse_args()

    args.model_name = os.path.basename(args.model_path)

    toker = AutoTokenizer.from_pretrained(args.model_path)
    TEMPLATE = open('./templates/critique_template.txt').read().strip()

    llm = LLM(
        model=args.model_path, tokenizer=args.model_path,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True, swap_space=16,
        max_num_seqs=20,
    )

    if not args.use_voting:
        sampling_params = SamplingParams(temperature=0.,
                                         max_tokens=32768 if 'QwQ' in args.model_path else 8192, seed=42)
    else:
        if 'Qwen2.5-Math' in args.model_path: # to ensure normal generation of Qwen2.5-Math-7B/72B-Instruct
            sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, n=args.voting_n,
                                            max_tokens=32768 if 'QwQ' in args.model_path else 8192, seed=42)
        elif "deepseek-r1" in args.model_path.lower() or "qwen3" in args.model_path.lower():
            sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, n=args.voting_n,
                                            max_tokens=32768, seed=42)
        else:
            sampling_params = SamplingParams(temperature=1, top_p=0.9, n=args.voting_n,
                                            max_tokens=32768 if 'QwQ' in args.model_path else 8192, seed=42)
    

    if args.configs is None:
        args.configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']

    for config in args.configs:
        if not args.use_voting:
            output_dir = os.path.join(args.output_dir, args.model_name)
        else:
            output_dir = os.path.join(args.output_dir, f'{args.model_name}_voting')
        os.makedirs(output_dir, exist_ok=True)

        input_data = load_dataset('json', data_files=f"./dataset/{config}.json")['train']
        prompt_token_ids = [apply_chat_template(toker, prepare_input_boxed(TEMPLATE, e))
                            for e in input_data]

        generations = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

        res_data = []
        for i in range(len(input_data)):
            d = input_data[i].copy()
            # 增加唯一 id 字段（可以根据需要自定义生成方式，这里直接使用索引）
            d["id"] = f"{config}_{i}"
            if not args.use_voting:
                generated_critique = generations[i].outputs[0].text
            else:
                generated_critique = [ee.text for ee in generations[i].outputs]
            d['generated_critique'] = generated_critique
            res_data.append(d)

        # 将生成结果保存到文件（JSONL 格式）
        output_file = os.path.join(output_dir, f"{config}_generation.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for e in res_data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"生成结果已保存：{output_file}")

if __name__ == '__main__':
    main()
