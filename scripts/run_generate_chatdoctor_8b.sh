export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
#export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 run_src/do_raise.py \
    --dataset_name ChatDoctor \
    --test_json_filename icliniq_filtered \
    --model_ckpt meta-llama/Meta-Llama-3.1-8B-Instruct \
    --note default \
    --num_rollouts 1 \
    --mode run \
    --disable_a5 \
    --disable_a8 \
    --disable_a7 \
    --disable_a6 \
    --enable_self_reward true \
#    --retrieval_corpus "textstat" \
#    --retrieval_threshold 0.5 \
#    --disable_a6 \
#    --disable_a1 \
#    --disable_a4 \
#    --disable_a3 \
#        --disable_a7 \
#    --disable_a1 \
#    --disable_a4 \
#    --disable_a3 \
#    --enable_answer_checking true \
#    --combine_distributions "disagreement" \
#    --search_query_weight 0.5 \
#    --enable_majority true \
#    --combine_distributions "multi" \
#    --combine_distributions "subtract" \
#    --majority_threshold 0.4
#    --disable_a1 \
#    --disable_a4 \
#    --disable_a3 \
#    --disable_a8 \
#    --disable_a6 \
#    --disable_a7
#    --disable_answer_selection \
#    --api llama \
#        --model_ckpt llama31_8b_with_openai_api \
#    --tensor_parallel_size 4
