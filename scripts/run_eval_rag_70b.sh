export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5 # 0,3 # 6,7 #4,5,6,7 # 4,5,6,7 # 0,1,2,3 #4,5,6,7 #,0,1,2,3 # 4,5,6,7
python3 run_src/eval_rag_batch_zs.py \
    --eval_model_name /data/experiment_data_gamma/hieutran/Meta-Llama-3.1-70B-Instruct  \
    --eval_data_path /home/htran/generation/med_preferences/raise/data \
    --eval_batch_size 4 \
    --eval_split valid \
    --eval_sample_size -1 \
    --num_shot 0 \
    --eval_save_path save/piqa_70b_rag.json \
    --eval_task PIQA \
    --max_new_tokens 16 \
#    /data/experiment_data_gamma/hieutran/Meta-Llama-3.1-70B-Instruct \
#        --eval_save_path /mnt/interns/hieutran/research/llm_safe/long-form-factuality/results/usmle_test_zs_evaluation.json \
#    --memory_0=80 \
#    --memory_1=80 \
#    --memory_2=70 \
#    --memory_3=40 \