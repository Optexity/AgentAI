#!/bin/bash
#SBATCH --job-name=llamma3_train
#SBATCH --output=/data/user_data/sachingo/Reinforce-Align-AI/slurm_outputs/%x/%A_%a.out  # Standard output
#SBATCH --error=/data/user_data/sachingo/Reinforce-Align-AI/slurm_outputs/%x/%A_%a.err   # Standard error
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=40G
#SBATCH --time=5:00:00
#SBATCH --priority=1
#SBATCH --array=0-0

cd /data/user_data/sachingo/Reinforce-Align-AI
source ~/.bashrc
conda activate browsergym

cd LLaMA-Factory
llamafactory-cli train ../AgentAI/agentai/train_configs/llama3.1_lora_sft_service_catalog.yaml

echo "Done"
