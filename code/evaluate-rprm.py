import os
import json
import numpy as np
from datasets import load_dataset
import argparse

# os.chdir(os.path.dirname(__file__))
print("当前代码运行所在的路径为：", os.getcwd())

def compute_match(responses, threshold=0.5):
    """
    计算responses中第一个小于0.2的索引，如果不存在则返回-1。
    """
    for idx, value in enumerate(responses):
        if value < threshold:
            return idx
    return -1

def process_dataset(input_dataset, config, sample_number=10, threshold=0.5):
    """
    处理数据集，计算match值，分类数据，并计算acc1、acc2和f1分数。

    参数：
    - input_dataset: datasets.Dataset对象
    - config: 配置名称，用于输出文件命名
    - output_dir: 输出文件夹名称（默认是'output'）
    """
    # 创建输出文件夹（如果不存在）
    # os.makedirs(output_dir, exist_ok=True)
    
    res_data = []
    token_number_total = 0
    # 遍历数据集并计算'match'值
    for data in input_dataset:
        # 计算'match'值
        now_response = data.get('responses', [])
        now_response = [item[:sample_number] for item in now_response]
        now_token_number = data.get('token_numbers', [])
        now_token_number = [item[:sample_number] for item in now_token_number]
        now_token_number = [sum(item) for item in now_token_number]
        # token_number_total += sum(now_token_number)
        # 每一行取平均值
        now_response = [sum(item) / len(item) for item in now_response]
        match = compute_match(now_response, threshold)
        if match == -1:
            token_number_total += sum(now_token_number)
        else:
            token_number_total += sum(now_token_number[: match])
        # 将'match'添加到数据中
        data_dict = dict(data)
        match = 1 if match == data.get('label', -1) else 0
        data_dict['match'] = match
        res_data.append(data_dict)
    
    # 根据'label'分类数据
    error_data = [e for e in res_data if e.get('label', -1) != -1]
    correct_data = [e for e in res_data if e.get('label', -1) == -1]
    
    
    # 计算acc1和acc2
    acc1 = np.mean([e['match'] for e in error_data]) * 100 if error_data else 0
    acc2 = np.mean([e['match'] for e in correct_data]) * 100 if correct_data else 0
    
    # 计算f1分数
    if acc1 + acc2 == 0:
        f1 = 0
    else:
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
    
    # 打印结果
    print(f'Sample Number: {sample_number} {config} f1: {f1:.1f} average token number: {token_number_total / len(input_dataset):.1f}')
    return f1, token_number_total / len(input_dataset)
    # 将结果保存到JSON文件
    # result = {
    #     'config': config,
    #     'error_acc': acc1,
    #     'correct_acc': acc2,
    #     'f1': f1
    # }
    
    # result_file = os.path.join(output_dir, f'result.json')
    # with open(result_file, 'w', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    dataset_path = args.dataset_path + "_origin.json"
    dataset = load_dataset(
        "json", 
        data_files=dataset_path, 
        split="train"
    )
    item = {
        "data_name": args.data_name,
        "result": []
    }
    result = []
    for sample_number in [1, 2, 4]:
        f1, token_number = process_dataset(dataset, args.data_name, sample_number=sample_number, threshold=0.5)
        item["result"].append({
            "sample_number": sample_number,
            "f1": f1,
            "token_number": token_number
        })
    result.append(item)
    output_path = args.output_path + "_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
    
