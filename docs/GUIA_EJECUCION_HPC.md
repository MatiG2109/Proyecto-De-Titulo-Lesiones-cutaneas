# Guía de ejecución de los experimentos en el clúster HPC (SLURM)

Esta guía describe cómo se ejecutaron los experimentos de regularización y optimización (E2–E6) en el clúster Océano (PUCV). Adapta rutas, partición y entorno a tu clúster.

## 1. Archivos necesarios

En la carpeta del proyecto del clúster (por ejemplo `/work/<usuario>/proyecto_tesis/`) deben estar:

- `preprocess.py`, `hybrid_model.py`, `train.py` y `train_v2.py`.
- La carpeta `sbatch/` con los scripts `.sbatch` de cada experimento (E2, E3, E4, E5a, E5b, E6).
- El dataset organizado en una carpeta por clase (ver el README principal).

`train_v2.py` es un script independiente: no modifica `train.py`, y con sus parámetros por defecto entrena igual que este.

## 2. Prueba rápida antes de lanzar (opcional, recomendada)

Con un subconjunto pequeño del dataset (`dataset_sample`, con la misma estructura de carpetas por clase):

```bash
python train_v2.py --data-dir dataset_sample --epochs 2 --batch-size 4
```

Si corre sin errores, se puede lanzar en el clúster.

## 3. Orden de ejecución

Lanza **un trabajo a la vez**. Al ejecutar dos en la misma GPU se produjo un error de memoria (`CUDA out of memory`). Espera a que cada uno termine antes de lanzar el siguiente.

```bash
cd /work/<usuario>/proyecto_tesis

sbatch sbatch/train_e2_earlystop.sbatch     # E2: early stopping
sbatch sbatch/train_e3_scheduler.sbatch     # E3: scheduler coseno
sbatch sbatch/train_e4_dropout.sbatch       # E4: dropout 0.5
sbatch sbatch/train_e5_bs16.sbatch          # E5a: tamaño de lote 16
sbatch sbatch/train_e5_bs64.sbatch          # E5b: tamaño de lote 64
sbatch sbatch/train_e6_focal.sbatch         # E6: focal loss

squeue -u $USER                             # estado de los trabajos
```

## 4. Resultados que deja cada trabajo

- **Log completo:** `logs/<jobid>_lesiones_e*.out`, con las métricas finales sobre el conjunto de test.
- **Checkpoint:** `checkpoints/best_model_e*.pt`.
- **Curvas por época:** `logs/best_model_e*_history.csv`, con `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_f1_macro` y `lr` por época, útil para graficar.

## 5. Experimentos posteriores

Con los resultados de E2–E6 (exactitud y F1 de test) se comparan las técnicas frente a la línea base (híbrido con fine-tuning completo: 82,58 % de exactitud) y se arma el experimento **E7**, que combina las mejores (lote de 16 + scheduler coseno). Finalmente se repite la mejor configuración (E5a) con distintas semillas de entrenamiento (`--train-seed`, script `sbatch/train_e5a_repeticiones.sbatch`) y se resume con:

```bash
python resumen_repeticiones.py --incluir-original
```

## 6. Exportación del modelo a ONNX

```bash
sbatch sbatch/export_onnx.sbatch    # genera skincancer.onnx y verifica la salida contra PyTorch
```
