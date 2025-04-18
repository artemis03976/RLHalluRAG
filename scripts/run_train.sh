export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --num_processes 2 train.py \
 --base_model_name "Qwen/Qwen2.5-32B-Instruct" \
 --use_hf_model \
 --evaluator_name "deepseek-v3-250324" \
 --method reinforce \
 --n_samples 20 \