#!/bin/bash
#SBATCH --job-name=llamma3_serving
#SBATCH --output=/data/user_data/sachingo/optexity/slurm_outputs/%x/%A_%a.out  # Standard output
#SBATCH --error=/data/user_data/sachingo/optexity/slurm_outputs/%x/%A_%a.err   # Standard error
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --priority=1
#SBATCH --array=0-8

cd /data/user_data/sachingo/optexity
source ~/.bashrc
conda activate browsergym

export PORT=673$SLURM_ARRAY_TASK_ID

cd LLaMA-Factory
echo "Starting LLaMA API server on port $PORT"
API_PORT=$PORT llamafactory-cli api ../AgentAI/agentai/inference_configs/llama3.1_lora_sft_service_catalog_tasks.yaml &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

sleep 2m

export SNOW_INSTANCE_URL="https://dev283325.service-now.com/"
export SNOW_INSTANCE_UNAME="admin"
export SNOW_INSTANCE_PWD="wx%h/z5WWW0J"

cd ../AgentAI/agentai
for seed in {0..9}; do
    echo "Starting inference for seed $seed"
    python main.py --seed $seed --task_num $SLURM_ARRAY_TASK_ID --port $PORT --headless --log_path ./logs/lora/service_catalog_tasks/
done

echo "Killing server"
kill -9 $SERVER_PID
echo "Done"
