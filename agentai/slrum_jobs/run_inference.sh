#!/bin/bash
#SBATCH --job-name=lamma3_serving
#SBATCH --output=/data/user_data/sachingo/Reinforce-Align-AI/slurm_outputs/%x/%A_%a.out  # Standard output
#SBATCH --error=/data/user_data/sachingo/Reinforce-Align-AI/slurm_outputs/%x/%A_%a.err   # Standard error
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=2G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END
#SBATCH --priority=1
#SBATCH --array=0-10

cd /data/user_data/sachingo/Reinforce-Align-AI
source ~/.bashrc
conda init
conda activate browsergym

python --version
echo "Running array task ID: $SLURM_ARRAY_TASK_ID"
echo "Done!"