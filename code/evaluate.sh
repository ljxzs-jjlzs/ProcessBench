
work_dir="" # 就是当前这个code文件夹路径，例如/home/nfs05/liujx/GithubRepos/ProcessBench/code
cd ${work_dir}
model_path=""
model_name=$(basename "$model_path")

python generate.py \
    --model_path $model_path \
    --configs gsm8k \
    --use_voting \
    --voting_n 32

python evaluate2.py \
    --input_file $(realpath ./outputs/${model_name}_voting/gsm8k_generation.jsonl) \
    --configs gsm8k \
    --output_dir $(realpath ./outputs)



python generate.py \
    --model_path $model_path \
    --configs math \
    --use_voting \
    --voting_n 32

python evaluate2.py \
    --input_file $(realpath ./outputs/${model_name}_voting/math_generation.jsonl) \
    --configs math \
    --output_dir $(realpath ./outputs)


python generate.py \
    --model_path $model_path \
    --configs olympiadbench \
    --use_voting \
    --voting_n 32

python evaluate2.py \
    --input_file $(realpath ./outputs/${model_name}_voting/olympiadbench_generation.jsonl) \
    --configs olympiadbench \
    --output_dir $(realpath ./outputs)

python generate.py \
    --model_path $model_path \
    --configs omnimath \
    --use_voting \
    --voting_n 32

python evaluate2.py \
    --input_file $(realpath ./outputs/${model_name}_voting/omnimath_generation.jsonl) \
    --configs omnimath \
    --output_dir $(realpath ./outputs)