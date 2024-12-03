export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
python3 run_src/do_raise.py \
    --dataset_name MMLU \
    --test_json_filename test_all \
    --model_ckpt meta-llama/Llama-3.2-3B-Instruct \
    --note default \
    --num_rollouts 1 \
    --mode run \
    --disable_a5 \
    --disable_a8 \
    --retrieval_corpus "textstat" \
    --retrieval_threshold 0.5 \
#    --num_queries 5 \
#        --half_precision \
#        --disable_a5 \
#    --disable_a7
#    --api llama \
#        --model_ckpt llama31_8b_with_openai_api \
#    --tensor_parallel_size 4
