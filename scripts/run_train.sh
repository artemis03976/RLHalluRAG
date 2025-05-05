export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=1,5,6,7

python train.py \
 --base_model_name "Qwen/Qwen2.5-32B-Instruct" \
 --use_hf_model \
 --evaluator_name "deepseek-v3-250324" \
 --method reinforce \
 --n_samples 1000 \
 --lr 0.0001 \
 --batch_size 4 \
 --n_gpus 4 \