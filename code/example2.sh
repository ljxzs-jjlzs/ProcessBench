work_dir="/home/nfs05/liujx/GithubRepos/ProcessBench/code"
output_dir="/home/nfs05/liujx/GithubRepos/ProcessBench/code/outputs/RPRM/Qwen2.5-0.5B-Instruct"
cd ${work_dir}
model_path="/home/nfs05/model/Qwen2.5-0.5B-Instruct"
config=gsm8k
model_name=$(basename $model_path)
echo $model_name
output_path=$output_dir/$config

python generate-rprm.py \
    --model_path $model_path \
    --configs $config \
    --output_path $output_path \
    --n 4 \
    --tensor_parallel_size 2

python return-to-origin.py \
    --tokenizer_path $model_path \
    --data_name $config \
    --response_path $output_path

python evaluate-rprm.py \
    --data_name $config \
    --dataset_path $output_path \
    --output_path $output_path
