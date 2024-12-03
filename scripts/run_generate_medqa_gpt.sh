export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
#export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 run_src/do_raise.py \
    --dataset_name MedQA \
    --test_json_filename test_all \
    --model_ckpt gpt-4o-mini \
    --api gpt3.5-turbo \
    --note default \
    --num_rollouts 2 \
    --mode run \
    --disable_a5 \
    --disable_a8 \
    --disable_a7 \
    --enable_chat_template True \
    --mcts_num_last_votes 10 \
    --retrieval_corpus "textstat" \
    --retrieval_threshold 0.5 \
#    --num_queries 3 \
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
