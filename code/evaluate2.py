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
import copy

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
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help="测试结果保存的目录")
    args = parser.parse_args()

    # 固定评测使用的 critique 数量列表
    numbers = [1, 2, 4, 8, 10, 16]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # 读取生成结果文件
    with open(args.input_file, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]

    # 存储所有评测结果
    eval_results = {}

    for num in numbers:
        token_numbers = 0
        res_data = []
        for d in all_data:
            # 为避免多次修改同一条数据，这里复制一份
            d_new = copy.deepcopy(d)
            generated = d_new.get("generated_critique", None)
            if generated is None:
                d_new["prediction"] = None
            else:
                # 如果生成结果为列表，则取前 num 个进行评估
                if isinstance(generated, list):
                    subset = generated[:num]
                    token_numbers += sum([len(tokenizer.encode(e)) for e in subset])
                    preds = [extract_answer(e) for e in subset]
                    preds = [e for e in preds if e is not None]
                    if preds:
                        d_new["prediction"] = Counter(preds).most_common(1)[0][0]
                        try:
                            d_new["prediction"] = int(d_new["prediction"])
                        except Exception:
                            d_new["prediction"] = None
                    else:
                        d_new["prediction"] = None
                else:
                    pred = extract_answer(generated)
                    token_numbers += len(tokenizer.encode(generated))
                    try:
                        d_new["prediction"] = int(pred)
                    except Exception:
                        d_new["prediction"] = None

            d_new["match"] = (d_new.get("prediction") == d_new.get("label"))
            res_data.append(d_new)

        # 根据 label 分离 error 和 correct 样本
        error_data = [e for e in res_data if e['label'] != -1]
        correct_data = [e for e in res_data if e['label'] == -1]

        # 计算准确率和 F1 分数
        acc1 = np.mean([e['match'] for e in error_data]) * 100 if error_data else 0.0
        acc2 = np.mean([e['match'] for e in correct_data]) * 100 if correct_data else 0.0
        f1 = 2 * acc1 * acc2 / (acc1 + acc2) if (acc1 + acc2) > 0 else 0.0

        eval_results[str(num)] = {
            "error_acc": round(acc1, 1),
            "correct_acc": round(acc2, 1),
            "f1": round(f1, 1),
            "average_token_numbers": round(token_numbers/len(all_data), 1)
        }
        print(f'{args.configs} using first {num} critique(s) -> error acc: {acc1:.1f}%, correct acc: {acc2:.1f}%, f1: {f1:.1f}%, average token numbers: {token_numbers/len(all_data):.1f}')

    # 创建输出目录并存储评测结果为 JSON 文件
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.configs}_evaluation.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=4, ensure_ascii=False)
    print(f"评测结果已保存到: {output_file}")

if __name__ == '__main__':
    main()
