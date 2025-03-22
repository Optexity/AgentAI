#!/bin/bash
#SBATCH --job-name=qwne_train
#SBATCH --output=/data/user_data/sachingo/Reinforce-Align-AI/slurm_outputs/%x/%A_%a.out  # Standard output
#SBATCH --error=/data/user_data/sachingo/Reinforce-Align-AI/slurm_outputs/%x/%A_%a.err   # Standard error
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=100G
#SBATCH --time=5:00:00
#SBATCH --priority=1
#SBATCH --array=0-0

cd /data/user_data/sachingo/Reinforce-Align-AI
source ~/.bashrc
conda activate browsergym

cd LLaMA-Factory
export WANDB_API_KEY="ac9187c6c3c683b2a0b2b5193b5e64fb403bdc40";
llamafactory-cli train data/train_data/hubspot_agent/Qwen/Qwen2.5-7B-Instruct-1M/train_config2.yaml

echo "Done"
