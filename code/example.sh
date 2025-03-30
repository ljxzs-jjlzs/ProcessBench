
work_dir=""
cd ${work_dir}
model_path=""
# CUDA_VISIBLE_DEVICES=0,1,6,7 python /home/nfs05/liujx/GithubRepos/ProcessBench/code/generate.py \
#     --model_path /home/nfs05/model/Qwen2.5-7B-Instruct \
#     --configs gsm8k \
#     --use_voting \
#     --voting_n 2

python run_eval.py \
    --model_path $model_path \
    --configs gsm8k math olympiadbench omnimath \
    --use_voting \
    --voting_n 16