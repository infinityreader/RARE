export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0 # 0,3 # 6,7 #4,5,6,7 # 4,5,6,7 # 0,1,2,3 #4,5,6,7 #,0,1,2,3 # 4,5,6,7
python3 run_src/eval_generation_api.py \
    --model_ckpt gpt-4o-mini \
    --eval_data_path /home/htran/generation/med_preferences/raise/data/ChatDoctor/icliniq_filtered.json \
    --eval_batch_size 1 \
    --eval_split test \
    --eval_sample_size 200 \
    --eval_save_path save/gpt-4o-mini-icliniq.json \
    --eval_task ChatDoctor \
    --max_new_tokens 512