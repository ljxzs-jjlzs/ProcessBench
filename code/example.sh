
work_dir=/home/nfs05/liujx/GithubRepos/ProcessBench/code
cd ${work_dir}

# CUDA_VISIBLE_DEVICES=0,1,6,7 python /home/nfs05/liujx/GithubRepos/ProcessBench/code/generate.py \
#     --model_path /home/nfs05/model/Qwen2.5-7B-Instruct \
#     --configs gsm8k \
#     --use_voting \
#     --voting_n 2

CUDA_VISIBLE_DEVICES=0,1,6,7 python /home/nfs05/liujx/GithubRepos/ProcessBench/code/run_eval.py \
    --model_path /home/nfs05/model/Qwen2.5-7B-Instruct \
    --configs gsm8k \
    --use_voting \
    --voting_n 2