export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 -m vllm.entrypoints.openai.api_server \
--model /home/htran/llama/Meta-Llama-3.1-70B-Instruct \
--gpu-memory-utilization 0.9 --tensor-parallel-size 2 \
--served-model-name llama31_7b_with_openai_api \
--max-model-len 32000 \
--tokenizer-pool-size 128 --swap-space 8 --block-size 32 \
--port 5999