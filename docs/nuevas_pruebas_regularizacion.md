# Plan de experimentos: regularización y optimización

## Contexto

Modelo híbrido ConvNeXt-tiny + Swin-tiny (`timm`, preentrenados en ImageNet), fusión por concatenación y cabezal `Linear(1536→256) → ReLU → Dropout → Linear(256→8)`, sobre ISIC 2019 (8 clases, fuerte desbalance: NV ≈ 50 % de las imágenes).

Punto de partida (fine-tuning completo, `lr = 1e-5`, sin scheduler, sin early stopping, dropout 0,3):

| Modelo | Exactitud | F1 ponderado |
|---|---|---|
| Híbrido | 82,58 % | 0,83 |
| ConvNeXt solo | 85,34 % | 0,85 |
| Swin solo | 82,21 % | 0,82 |

## Objetivo

Evaluar el efecto individual de distintas técnicas de regularización y optimización sobre el modelo híbrido con el backbone descongelado, ejecutando los experimentos de a uno. Para ello se creó `train_v2.py`, que incorpora las técnicas como parámetros opcionales sin modificar `train.py` ni los checkpoints ya generados.

## Técnicas disponibles en `train_v2.py`

Todas están desactivadas por defecto. Con los valores por defecto el entrenamiento coincide con el de `train.py` (dropout 0,3, weight decay 0,01, sin scheduler, sin early stopping), con una diferencia: `train_v2.py` guarda el checkpoint según el **mayor F1 macro de validación**, mientras que `train.py` lo hace según la menor pérdida de validación.

| Técnica | Parámetro | Por defecto |
|---|---|---|
| Dropout ajustable | `--dropout-rate` | 0,3 |
| Early stopping (monitorea F1 macro de validación) | `--early-stopping-patience` | desactivado |
| Scheduler de tasa de aprendizaje | `--scheduler {none,cosine,plateau,onecycle}` | `none` |
| Tamaño de lote | `--batch-size` | 4 (en los experimentos se usó 16, 32 y 64) |
| Label smoothing | `--label-smoothing` | 0,0 |
| Focal loss (conserva los pesos por clase) | `--use-focal-loss`, `--focal-gamma` | desactivada, γ = 2,0 |
| Precisión mixta (AMP, solo GPU) | `--amp` | desactivada |
| Gradient clipping | `--grad-clip` | desactivado |
| Weight decay | `--weight-decay` | 0,01 |
| Semillas | `--seed` (partición), `--train-seed` (entrenamiento), `--deterministic` | 42, no fijada |

Además, cada entrenamiento guarda en `logs/<checkpoint>_history.csv` el historial por época (`train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_f1_macro`, `lr`), útil para graficar las curvas de aprendizaje, y al final entrega las métricas de test (exactitud, precisión, recall y F1 por clase, promedios macro y ponderado, y matriz de confusión).

## Experimentos realizados

Todos con fine-tuning completo (`--no-freeze-backbone --lr 1e-5 --epochs 30`).

| Exp. | Técnica | Configuración | Checkpoint | Exactitud en test |
|---|---|---|---|---|
| E1 | Referencia (línea base) | fine-tuning completo | `best_model_finetune.pt` | 82,58 % |
| E2 | Early stopping | `--early-stopping-patience 5` | `best_model_e2_earlystop.pt` | 85,87 % |
| E3 | Scheduler coseno | `--scheduler cosine` | `best_model_e3_cosine.pt` | 86,32 % |
| E4 | Dropout aumentado | `--dropout-rate 0.5` | `best_model_e4_dropout.pt` | 84,87 % |
| E5a | Lote de 16 | `--batch-size 16` | `best_model_e5_bs16.pt` | **87,16 %** |
| E5b | Lote de 64 | `--batch-size 64` | `best_model_e5_bs64.pt` | 86,21 % |
| E6 | Focal loss | `--use-focal-loss --focal-gamma 2.0` | `best_model_e6_focal.pt` | 78,29 % |
| E7 | Combinado (lote 16 + coseno) | `--batch-size 16 --scheduler cosine` | `best_model_e7_combinado.pt` | 86,92 % |

Observaciones:

- La mejor configuración individual fue E5a (lote de 16). Combinar técnicas (E7) no la superó, es decir, los efectos no son aditivos.
- La focal loss (E6) empeoró el resultado respecto a la línea base.
- E5a se repitió con cinco corridas (la original más cuatro con `--train-seed 1` a `4`): **86,42 % ± 0,71** de exactitud. Resumen con `python resumen_repeticiones.py --incluir-original`.
- Parte de la mejora respecto a la línea base se debe al cambio del criterio de selección del checkpoint (de menor pérdida de validación a mayor F1 macro de validación), por lo que las comparaciones con la línea base deben leerse con cautela.

El orden de ejecución en el clúster está en [`GUIA_EJECUCION_HPC.md`](GUIA_EJECUCION_HPC.md).
