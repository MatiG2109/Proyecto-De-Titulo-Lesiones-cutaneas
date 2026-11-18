#!/bin/bash
#SBATCH --job-name=export_onnx
#SBATCH --partition=GPU
#SBATCH --qos=normal-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/work/matias.gallardo/proyecto_tesis/logs/%j_%x.out
#SBATCH --error=/work/matias.gallardo/proyecto_tesis/logs/%j_%x.err

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate lesiones

cd /work/matias.gallardo/proyecto_tesis

python export_onnx.py /work/matias.gallardo/dataset/AK/ISIC_0024468.jpg
ls -lh skincancer.onnx
