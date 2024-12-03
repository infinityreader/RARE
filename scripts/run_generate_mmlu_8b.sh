export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
#export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 run_src/do_raise.py \
    --dataset_name MMLU \
    --test_json_filename test_all \
    --model_ckpt /data/data_user_alpha/public_models/Llama-3.1/Meta-Llama-3.1-8B-Instruct \
    --note default \
    --num_rollouts 1 \
    --mode run \
    --disable_a5 \
    --disable_a8 \
    --disable_a6 \
    --disable_a7 \
    --enable_chat_template true
#    --retrieval_corpus "textstat" \
#    --retrieval_threshold 0.5 \
#        --disable_a6 \
#    --enable_chat_template
#    --num_queries 3 \
#    --num_retrieval 3 \
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
