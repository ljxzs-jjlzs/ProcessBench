import argparse
import numpy as np
import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

def apply_chat_template(toker, messages):
    input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return input_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        choices=['gsm8k', 'math', 'olympiadbench', 'omnimath'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--output_path", type=str, default='./outputs')
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--tensor_parallel_size', type=int, default=4)
    args = parser.parse_args()

    args.model_name = os.path.basename(args.model_path)

    toker = AutoTokenizer.from_pretrained(args.model_path)
    TEMPLATE = open('./templates/r-prm-prompt.txt').read().strip()

    llm = LLM(
        model=args.model_path, tokenizer=args.model_path,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True, swap_space=16,
        max_num_seqs=20,
    )

    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, n=args.n,
                                            max_tokens=32768, seed=42)


    for config in args.configs:
        # output_dir = args.output_dir
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        input_data = load_dataset('json', data_files=f"./dataset/{config}_filtered.json")['train']
        querys = []
        for data in input_data:
            query = TEMPLATE.format(data['question'], data['previous_steps'], data['now_step'])
            messages = [{'role': 'user', 'content': query}]
            querys.append(messages)
        prompts = [apply_chat_template(toker, query) for query in querys]
        print("=================================================")
        print(prompts[0])
        print("=================================================")
        generations = llm.generate(prompts, sampling_params=sampling_params)

        res_data = []
        for i in range(len(input_data)):
            d = input_data[i].copy()
            # 增加唯一 id 字段（可以根据需要自定义生成方式，这里直接使用索引）
            # d["id"] = f"{d['origin_id']}_{i}"
            generated_critique = [ee.text for ee in generations[i].outputs]
            d['generated_critique'] = generated_critique
            judges = []
            for generated_critique in generated_critique:
                if "Is the step correct (Yes/No)?" not in generated_critique:
                    judges.append(0)
                else:
                    contents = generated_critique.split("Is the step correct (Yes/No)?")[1].strip()
                    if contents.startswith("Yes"):
                        judges.append(1)
                    else:
                        judges.append(0)
            d['judges'] = judges
            res_data.append(d)
        output_path = args.output_path + "_generation.jsonl"
        # output_file = os.path.join(os.path.dirname(args.output_path), f"{config}_generation.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for e in res_data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"生成结果已保存：{output_path}")

if __name__ == '__main__':
    main()
