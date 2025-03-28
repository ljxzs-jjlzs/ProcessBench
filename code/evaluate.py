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

def extract_answer(solution_text: str):
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help="生成结果的文件路径（JSONL 格式，每行包含一个样本信息，样本中需包含 config 字段）")
    parser.add_argument('--configs', type=str, required=True,
                        help="待评测的配置名称，例如: gsm8k 或 math")
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help="测试结果保存的目录")
    parser.add_argument('--number', type=int, default=None,
                        help="当生成结果为列表时，只使用前 number 个 critique 进行计算")
    args = parser.parse_args()

    # 读取生成结果文件
    with open(args.input_file, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]

    # 根据样本中的 config 字段筛选记录
    records = all_data
    res_data = []
    for d in records:
        generated = d.get("generated_critique", None)
        if generated is None:
            d["prediction"] = None
        else:
            # 针对投票情况（列表）和普通情况（字符串）分别处理
            if isinstance(generated, list):
                generated = generated[:args.number] if args.number else generated
                preds = [extract_answer(e) for e in generated]
                preds = [e for e in preds if e is not None]
                if preds:
                    d["prediction"] = Counter(preds).most_common(1)[0][0]
                    try:
                        d["prediction"] = int(d["prediction"])
                    except Exception:
                        d["prediction"] = None
                else:
                    d["prediction"] = None
            else:
                pred = extract_answer(generated)
                try:
                    d["prediction"] = int(pred)
                except Exception:
                    d["prediction"] = None

        d["match"] = (d.get("prediction") == d.get("label"))
        res_data.append(d)

    # 根据 label 分离 error 和 correct 样本
    error_data = [e for e in res_data if e['label'] != -1]
    correct_data = [e for e in res_data if e['label'] == -1]

    # 计算准确率和 F1 分数
    acc1 = np.mean([e['match'] for e in error_data]) * 100
    acc2 = np.mean([e['match'] for e in correct_data]) * 100
    f1 = 2 * acc1 * acc2 / (acc1 + acc2)
    print(f'{args.configs} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')

if __name__ == '__main__':
    main()
