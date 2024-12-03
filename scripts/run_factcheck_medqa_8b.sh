export CUDA_VISIBLE_DEVICES=0
python3 fact_check_analysis.py \
    --num_votes 5 \
    --rate_model_name gpt-4o-mini \
    --suffix "bool"  \
    --data_file_path /home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_ro1_s5_r3_textstat_t5_a1_a3_a4_a6_fm.json \
#    --data_file_path /home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_8_a1_a6_a8.json \
#        --data_file_path /home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_8_a1_a6_a8.json  \
#        --data_file_path /home/htran/generation/med_preferences/raise/save/Meta-Llama-3.1-8B-Instruct_MedQA_4_a1_a4_a6.json  \