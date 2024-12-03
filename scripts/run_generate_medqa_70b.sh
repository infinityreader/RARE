export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 run_src/do_raise.py \
    --dataset_name SIQA \
    --test_json_filename dev \
    --model_ckpt /data/experiment_data_gamma/hieutran/Meta-Llama-3.1-70B-Instruct \
    --note default \
    --num_rollouts 1 \
    --mode run \
    --disable_a5 \
    --disable_a8 \
    --tensor_parallel_size 2 \
    --mcts_num_last_votes 32 \
    --enable_chat_template true \
    --disable_a7 \
    --disable_a4 \
    --retrieval_corpus "wikipedia" \
    --retrieval_threshold 0.5 \
#    --num_queries 5 \
#    --num_retrieval 3 \
#        --disable_a6 \
#    --enable_chat_template
#    --retrieval_corpus "medcorp" \
#    --retrieval_threshold 0.5 \
#    --enable_answer_checking true \
#    --combine_distributions "disagreement" \
#    --search_query_weight 0.5 \
#    --enable_majority true \
#    --combine_distributions "multi" \
#    --combine_distributions "subtract" \
#    --majority_threshold 0.4
#    --disable_a8 \
#    --disable_a6 \
#    --disable_a7
#    --disable_answer_selection \
#    --api llama \
#        --model_ckpt llama31_8b_with_openai_api \
#    --tensor_parallel_size 4
