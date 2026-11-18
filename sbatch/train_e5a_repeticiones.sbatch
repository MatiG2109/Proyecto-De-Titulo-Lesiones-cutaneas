#!/bin/bash
#SBATCH --job-name=lesiones_e5a_rep
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

# Repeticiones de la mejor configuracion (E5a: fine-tuning, batch size 16, lr 1e-5)
# con distintas semillas de ENTRENAMIENTO. El split estratificado queda fijo en
# --seed 42 (por defecto), asi el conjunto de test es el mismo en todas las corridas.
# Las 4 corridas se ejecutan EN SECUENCIA dentro de este unico job (no en paralelo).

for SEED in 1 2 3 4; do
  echo "=== Repeticion E5a: train-seed=${SEED} ==="
  python3 -u train_v2.py \
    --data-dir /work/matias.gallardo/dataset \
    --epochs 30 \
    --batch-size 16 \
    --num-workers 8 \
    --lr 1e-5 \
    --no-freeze-backbone \
    --seed 42 \
    --train-seed ${SEED} \
    --checkpoint-path checkpoints/best_model_e5a_seed${SEED}.pt \
    2>&1 | tee logs/e5a_seed${SEED}.log
done

echo ""
echo "=== Resumen de las repeticiones ==="
grep -h RESULTADO_FINAL logs/e5a_seed*.log
