#!/bin/bash
#SBATCH --job-name=medqa_raise
#SBATCH --gres=gpu:1
#SBATCH --time=150:00:00
#SBATCH --partition=gypsum-rtx8000
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --output=./medqa_raise_%j.out
#SBATCH --error=./medqa_raise_%j.error
python3 run_src/do_raise.py \
    --dataset_name MedQA \
    --test_json_filename test_all \
    --model_ckpt meta-llama/Llama-3.2-3B-Instruct \
    --note default \
    --num_rollouts 4 \
    --mode run \
    --half_precision \
    --disable_a5 \
    --disable_a6 \
    --disable_a7

#        --disable_a5 \
#    --disable_a7
#    --api llama \
#        --model_ckpt llama31_8b_with_openai_api \
#    --tensor_parallel_size 4
