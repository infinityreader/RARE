export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2 # 0,3 # 6,7 #4,5,6,7 # 4,5,6,7 # 0,1,2,3 #4,5,6,7 #,0,1,2,3 # 4,5,6,7
python3 run_src/eval_cot_batch.py \
    --eval_model_name meta-llama/Llama-3.2-3B-Instruct \
    --eval_data_path /home/htran/generation/med_preferences/raise/data \
    --eval_batch_size 8 \
    --eval_split test_all \
    --eval_sample_size -1 \
    --num_shot 3 \
    --eval_save_path save/MMLU_3b_3s.json \
    --eval_task MMLU \
    --max_new_tokens 256 \
#        --eval_save_path /mnt/interns/hieutran/research/llm_safe/long-form-factuality/results/usmle_test_zs_evaluation.json \
#    --memory_0=80 \
#    --memory_1=80 \
#    --memory_2=70 \
#    --memory_3=40 \