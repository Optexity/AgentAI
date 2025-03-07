#!/bin/bash
#SBATCH --job-name=llamma3_serving
#SBATCH --output=/data/user_data/sachingo/Reinforce-Align-AI/slurm_outputs/%x/%A_%a.out  # Standard output
#SBATCH --error=/data/user_data/sachingo/Reinforce-Align-AI/slurm_outputs/%x/%A_%a.err   # Standard error
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --priority=1
#SBATCH --array=0-1

cd /data/user_data/sachingo/Reinforce-Align-AI
source ~/.bashrc
conda activate browsergym

export PORT=800$SLURM_ARRAY_TASK_ID
API_PORT=$PORT llamafactory-cli api AgentAI/agentai/inference_configs/llama3.1_lora_sft_service_catalog.yaml &
SERVER_PID=$!

sleep 2m

cd AgentAI/agentai
for seed in {0..9}; do
    python main.py --seed $seed --task_num $SLURM_ARRAY_TASK_ID --port $PORT --headless
done

kill -9 $SERVER_PID
