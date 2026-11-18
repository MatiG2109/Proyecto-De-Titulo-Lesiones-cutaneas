#!/bin/bash
#SBATCH --job-name=lesiones_e5_bs16
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

python train_v2.py \
  --data-dir /work/matias.gallardo/dataset \
  --epochs 30 \
  --batch-size 16 \
  --num-workers 8 \
  --lr 1e-5 \
  --no-freeze-backbone \
  --checkpoint-path checkpoints/best_model_e5_bs16.pt
