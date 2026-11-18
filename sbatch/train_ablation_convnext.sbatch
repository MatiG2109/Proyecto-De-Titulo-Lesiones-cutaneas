#!/bin/bash
#SBATCH --job-name=lesiones_ablation_convnext
#SBATCH --partition=GPU
#SBATCH --qos=normal-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/work/matias.gallardo/proyecto_tesis/logs/%j_%x.out
#SBATCH --error=/work/matias.gallardo/proyecto_tesis/logs/%j_%x.err

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate lesiones

cd /work/matias.gallardo/proyecto_tesis

python train_ablation.py \
  --data-dir /work/matias.gallardo/dataset \
  --backbone convnext \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 8 \
  --lr 1e-5 \
  --checkpoint-path checkpoints/best_model_convnext_only.pt
