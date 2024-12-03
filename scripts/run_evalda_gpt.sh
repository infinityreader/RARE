export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0 # 0,3 # 6,7 #4,5,6,7 # 4,5,6,7 # 0,1,2,3 #4,5,6,7 #,0,1,2,3 # 4,5,6,7
python3 run_src/eval_da_api.py \
    --eval_data_path /home/htran/generation/med_preferences/raise/data \
    --eval_batch_size 1 \
    --eval_sample_size -1 \
    --num_shot 3 \
    --max_new_tokens 16 \
    --eval_task MedQA \
    --eval_split test_all \
    --eval_save_path save/gpt-4o-mini-medqa-da-3s.json \
    --model_ckpt gpt-4o-mini \
#        --model_ckpt gpt-4o
#--eval_save_path save/claude-3-opus-stg-cot.json \