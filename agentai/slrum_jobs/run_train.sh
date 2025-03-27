#!/bin/bash
#SBATCH --job-name=llamma3_train
#SBATCH --output=/data/user_data/sachingo/optexity/slurm_outputs/%x/%A_%a.out  # Standard output
#SBATCH --error=/data/user_data/sachingo/optexity/slurm_outputs/%x/%A_%a.err   # Standard error
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=100G
#SBATCH --time=5:00:00
#SBATCH --priority=1
#SBATCH --array=0-0

cd /data/user_data/sachingo/optexity
source ~/.bashrc
conda activate browsergym

cd LLaMA-Factory
llamafactory-cli train data/train_data/hubspot_agent/meta-llama/Meta-Llama-3.1-8B-Instruct/train_config2.yaml

echo "Done"
