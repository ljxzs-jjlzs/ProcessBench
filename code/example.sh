
work_dir="/home/nfs05/liujx/GithubRepos/ProcessBench/code"
output_dir="/home/nfs05/liujx/GithubRepos/ProcessBench/code/outputs"
cd ${work_dir}
model_path="/home/nfs05/model/Qwen2.5-1.5B-Instruct"
model_name=$(basename $model_path)
echo $model_name

CUDA_VISIBLE_DEVICES=0,1,2,3 python generate.py \
    --model_path $model_path \
    --configs gsm8k math olympiadbench omnimath \
    --output_dir $output_dir \
    --use_voting \
    --voting_n 16 \
    --tensor_parallel_size 4

python evaluate2.py \
    --input_file $output_dir/${model_name}_voting/gsm8k_generation.jsonl \
    --configs gsm8k \
    --tokenizer_path $model_path \
    --output_dir $output_dir/${model_name}_voting 

python evaluate2.py \
    --input_file $output_dir/${model_name}_voting/math_generation.jsonl \
    --configs math \
    --tokenizer_path $model_path \
    --output_dir $output_dir/${model_name}_voting 

python evaluate2.py \
    --input_file $output_dir/${model_name}_voting/olympiadbench_generation.jsonl \
    --configs olympiadbench \
    --tokenizer_path $model_path \
    --output_dir $output_dir/${model_name}_voting 

python evaluate2.py \
    --input_file $output_dir/${model_name}_voting/omnimath_generation.jsonl \
    --configs omnimath \
    --tokenizer_path $model_path \
    --output_dir $output_dir/${model_name}_voting 