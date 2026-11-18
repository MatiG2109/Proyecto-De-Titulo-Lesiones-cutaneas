# Detección Automatizada de Lesiones Cutáneas con una arquitectura híbrida ConvNeXt + Swin Transformer

Proyecto de Título · Escuela de Ingeniería Informática · Pontificia Universidad Católica de Valparaíso (PUCV) · Noviembre 2026

**Autores:** Javier Ignacio Andaur Fariña · Matías Gerardo Gallardo Barrios
**Profesor guía:** Pamela Hermosilla · **Profesor correferente:** Emanuel Vega

Demo en el navegador: https://innovaprojects.cl/skincancer/

> **Aviso.** Este software es un prototipo académico de apoyo al prediagnóstico. **No reemplaza el diagnóstico ni el criterio clínico de un profesional de la salud** y no ha sido validado clínicamente ni evaluado por especialistas. Fue entrenado solo con imágenes dermatoscópicas y no es confiable con fotografías convencionales (por ejemplo, de celular).

---

## 1. Descripción

El melanoma requiere detección temprana, pero su evaluación visual es difícil porque existen lesiones benignas que se le parecen (por ejemplo, el nevus melanocítico). Este proyecto propone y evalúa un clasificador de **8 categorías diagnósticas** a partir de imágenes dermatoscópicas del dataset **ISIC 2019**:

| Sigla | Diagnóstico | Categoría de riesgo en la interfaz |
|---|---|---|
| MEL | Melanoma | Maligno |
| BCC | Carcinoma basocelular | Maligno |
| SCC | Carcinoma escamocelular | Maligno |
| AK | Queratosis actínica | Premaligno |
| NV | Nevus melanocítico | Benigno |
| BKL | Queratosis benigna | Benigno |
| DF | Dermatofibroma | Benigno |
| VASC | Lesión vascular | Benigno |

La propuesta combina dos redes en paralelo, ambas preentrenadas en ImageNet (biblioteca `timm`):

- **ConvNeXt-tiny**: extrae características locales y jerárquicas (bordes, texturas, color).
- **Swin Transformer-tiny**: aporta información contextual mediante atención por ventanas desplazadas.

Los vectores de ambas ramas (768 dimensiones cada uno) se fusionan por **concatenación simple** (1536 dimensiones) y pasan a un cabezal clasificador.

```
Imagen 224x224 ──┬─► ConvNeXt-tiny ──► 768 ─┐
                 │                           ├─ concat (1536) ─► Linear(1536→256) ─► ReLU ─► Dropout ─► Linear(256→8)
                 └─► Swin-tiny ──────► 768 ─┘
```

El estudio incluye transfer learning con backbone congelado, fine-tuning completo, un estudio de ablación (cada backbone por separado), experimentos de regularización y optimización, cinco repeticiones de la mejor configuración y un prototipo de interfaz que se ejecuta directamente en el navegador.

## 2. Resultados principales

Evaluación sobre el conjunto de test (3.800 imágenes; partición estratificada 70/15/15 con semilla 42).

| Modelo / experimento | Exactitud | F1 macro |
|---|---|---|
| Híbrido, backbone congelado | 67,05 % | 0,53 |
| Híbrido, fine-tuning completo (base) | 82,58 % | 0,77 |
| ConvNeXt-tiny solo (ablación) | 85,34 % | 0,81 |
| Swin-tiny solo (ablación) | 82,21 % | 0,76 |
| E2: early stopping | 85,87 % | 0,8154 |
| E3: scheduler coseno | 86,32 % | 0,8140 |
| E4: dropout 0,5 | 84,87 % | 0,8054 |
| **E5a: lote de 16** | **87,16 %** | 0,8173 |
| E5b: lote de 64 | 86,21 % | 0,8220 |
| E6: focal loss | 78,29 % | 0,7819 |
| E7: lote 16 + coseno | 86,92 % | 0,8176 |
| **E5a, 5 repeticiones** | **86,42 % ± 0,71** (rango 85,53–87,16) | 0,8127 ± 0,0087 |

Lectura de los resultados (detalle y limitaciones en el informe):

- El fine-tuning completo mejora de forma sustancial al backbone congelado.
- La evidencia respalda un desempeño **competitivo** del híbrido, pero **no que supere de forma concluyente** a ConvNeXt solo: ConvNeXt se entrenó una sola vez y con la configuración base, sin las técnicas de optimización aplicadas al híbrido.
- Combinar todas las técnicas (E7) no suma: el mejor resultado fue el de la técnica aislada (E5a).
- La literatura más cercana (MedFusionNet, ConvNeXt + ViT con fusión por atención) reporta 97,90 % en ISIC 2019; la diferencia sugiere que el mecanismo de fusión es un factor clave, y la concatenación simple es una limitación de esta propuesta.

![Comparación de exactitud](figuras/comparacion_accuracy.png)
![Matriz de confusión E5a](figuras/matriz_confusion_e5a.png)

## 3. Estructura del repositorio

```
.
├── preprocess.py            # dataset, partición estratificada, transformaciones, DataLoaders
├── hybrid_model.py          # arquitectura híbrida ConvNeXt + Swin
├── ablation_model.py        # modelos de un solo backbone (ablación)
├── train.py                 # entrenamiento base (congelado / fine-tuning)
├── train_v2.py              # entrenamiento con regularización y optimización (E2–E7, repeticiones)
├── train_ablation.py        # entrenamiento de los modelos de ablación
├── evaluate.py              # matriz de confusión y reporte sobre test desde un checkpoint
├── infer.py                 # inferencia sobre una imagen o una carpeta
├── resumen_repeticiones.py  # media y desviación estándar de las repeticiones de E5a
├── app.py                   # interfaz web local (Gradio)
├── export_onnx.py           # exporta el modelo a ONNX para el navegador
├── sbatch/                  # scripts SLURM usados en el clúster HPC Océano (PUCV)
├── logs/                    # historiales por época (CSV) y logs de las repeticiones
├── figuras/                 # matrices de confusión y gráficos de comparación
├── web/                     # sitio estático que ejecuta el modelo en el navegador
└── docs/                    # guías y notas de los experimentos
```

Los pesos entrenados (`*.pt`, 217 MB cada uno) y el modelo ONNX (216 MB) **no están en el repositorio** por superar el límite de GitHub; se publican en la sección **Releases** (ver sección 8).

## 4. Instalación

Requiere Python 3.11 (el usado en el proyecto) y, para entrenar, una GPU con CUDA.

```bash
git clone https://github.com/MatiG2109/Proyecto-De-Titulo-Lesiones-cutaneas.git
cd Proyecto-De-Titulo-Lesiones-cutaneas
python -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

En el clúster se utilizó un entorno conda (`lesiones`) con PyTorch 2.5.1 y CUDA 12.1.

## 5. Datos

El dataset **no se incluye**. Descarga ISIC 2019 (training, 25.331 imágenes; en el proyecto se obtuvo desde Kaggle) y organízalo en una carpeta por clase:

```
dataset/
  AK/    ISIC_0024468.jpg ...
  BCC/
  BKL/
  DF/
  MEL/
  NV/
  SCC/
  VASC/
```

Consulta la licencia y las condiciones de uso del dataset en el sitio del ISIC antes de redistribuirlo. Todas las rutas de ejemplo de este README usan `/ruta/a/dataset`; ajústalas a tu entorno.

## 6. Cómo funciona cada script

### `preprocess.py`: datos
- `scan_dataset(data_dir)`: recorre las carpetas por clase, abre cada imagen con PIL para verificarla y devuelve la lista `(ruta, clase)`. No detecta imágenes duplicadas.
- `stratified_split(samples, val_size, test_size, seed)`: partición estratificada en dos etapas con `train_test_split` (15 % test; luego 15 % del total para validación). Con la semilla 42 los conjuntos son siempre idénticos: 17.731 / 3.800 / 3.800 imágenes.
- `get_transforms(train)`: redimensiona a 224×224 y normaliza con media y desviación de ImageNet. Solo en entrenamiento añade volteos horizontal y vertical, rotación de ±20° y variaciones de brillo, contraste y saturación de ±20 %.
- `ISICDataset` y `build_dataloaders`: Dataset y DataLoaders de PyTorch para train, val y test.
- Prueba rápida: `python preprocess.py --data-dir /ruta/a/dataset --batch-size 8`

### `hybrid_model.py`: arquitectura
Define `HybridConvNeXtSwin(num_classes=8, pretrained=True, freeze_backbone=True, dropout_rate=0.3)`: dos backbones de `timm` (`convnext_tiny` y `swin_tiny_patch4_window7_224`), concatenación de sus características y el cabezal `Linear(1536→256) → ReLU → Dropout → Linear(256→8)`. Con `freeze_backbone=True` solo se entrena el cabezal (transfer learning); con `False`, todo el modelo (fine-tuning).
Prueba rápida: `python hybrid_model.py --batch-size 4` (usa un batch aleatorio si no se indica `--data-dir`).

### `train.py`: entrenamiento base
Cross-Entropy ponderada por clase (peso = N / (8 · n_clase)) y AdamW. Guarda el checkpoint con **menor pérdida de validación** y al terminar evalúa en test (exactitud, precisión, recall, F1 por clase y matriz de confusión).

```bash
# Transfer learning con backbone congelado (por defecto)
python train.py --data-dir /ruta/a/dataset --epochs 30 --batch-size 32 --num-workers 8 \
    --checkpoint-path checkpoints/best_model.pt

# Fine-tuning completo
python train.py --data-dir /ruta/a/dataset --epochs 30 --batch-size 32 --num-workers 8 \
    --lr 1e-5 --no-freeze-backbone --checkpoint-path checkpoints/best_model_finetune.pt
```

### `train_v2.py`: regularización y optimización
Versión extendida de `train.py`. Con todos los parámetros por defecto entrena igual que `train.py`, con una diferencia: **el checkpoint se elige por mayor F1 macro de validación**, no por pérdida. Parámetros: `--dropout-rate`, `--weight-decay`, `--early-stopping-patience`, `--scheduler {none,cosine,plateau,onecycle}`, `--label-smoothing`, `--use-focal-loss`, `--focal-gamma`, `--amp`, `--grad-clip`, `--history-csv`, y para reproducibilidad `--seed` (partición) y `--train-seed` (aleatoriedad del entrenamiento, junto con `--deterministic`). Guarda un CSV por época en `logs/<checkpoint>_history.csv` e imprime una línea `RESULTADO_FINAL` que luego lee `resumen_repeticiones.py`. El plan completo de experimentos (técnicas, configuración y resultados de E1 a E7) está en [`docs/nuevas_pruebas_regularizacion.md`](docs/nuevas_pruebas_regularizacion.md).

```bash
# E2: early stopping
python train_v2.py --data-dir /ruta/a/dataset --epochs 30 --no-freeze-backbone --lr 1e-5 \
    --early-stopping-patience 5 --checkpoint-path checkpoints/best_model_e2_earlystop.pt
# E3: scheduler coseno
python train_v2.py ... --scheduler cosine --checkpoint-path checkpoints/best_model_e3_cosine.pt
# E4: dropout 0.5
python train_v2.py ... --dropout-rate 0.5 --checkpoint-path checkpoints/best_model_e4_dropout.pt
# E5a / E5b: tamaño de lote 16 / 64
python train_v2.py ... --batch-size 16 --checkpoint-path checkpoints/best_model_e5_bs16.pt
python train_v2.py ... --batch-size 64 --checkpoint-path checkpoints/best_model_e5_bs64.pt
# E6: focal loss
python train_v2.py ... --use-focal-loss --focal-gamma 2.0 --checkpoint-path checkpoints/best_model_e6_focal.pt
# E7: combinado (lote 16 + coseno)
python train_v2.py ... --batch-size 16 --scheduler cosine --checkpoint-path checkpoints/best_model_e7_combinado.pt
```
(`...` = `--data-dir /ruta/a/dataset --epochs 30 --no-freeze-backbone --lr 1e-5`.) Los comandos exactos usados en el clúster están en `sbatch/`.

### `ablation_model.py` y `train_ablation.py`: estudio de ablación
`ablation_model.py` define `SingleBackboneModel` (un solo backbone + el mismo cabezal, sin fusión). `train_ablation.py` lo entrena con el mismo pipeline de datos, pesos de clase, optimizador y tasa de aprendizaje que el híbrido base, y guarda el checkpoint con **mayor exactitud de validación**.

```bash
python train_ablation.py --data-dir /ruta/a/dataset --backbone convnext --epochs 30 \
    --batch-size 32 --lr 1e-5 --checkpoint-path checkpoints/best_model_convnext_only.pt
python train_ablation.py --data-dir /ruta/a/dataset --backbone swin ... \
    --checkpoint-path checkpoints/best_model_swin_only.pt
```

### `resumen_repeticiones.py`: repeticiones de E5a
Lee las líneas `RESULTADO_FINAL` de `logs/e5a_seed*.log` y calcula media y desviación estándar muestral. `--incluir-original` suma la corrida original de E5a (87,16 %), que se ejecutó sin fijar la semilla de entrenamiento.
```bash
python resumen_repeticiones.py --incluir-original
```

### `evaluate.py`: matriz de confusión
Recrea la misma partición (misma semilla), evalúa un checkpoint sobre el test y guarda la matriz de confusión como PNG, además de imprimir el reporte de clasificación.
```bash
python evaluate.py --checkpoint checkpoints/best_model_e5_bs16.pt --data-dir /ruta/a/dataset \
    --out figuras/matriz_confusion_e5a.png --title "Matriz de Confusión — E5a (lote 16)"
```

### `infer.py`: inferencia por línea de comandos
```bash
python infer.py --checkpoint checkpoints/best_model_e5_bs16.pt --image ruta/a/imagen.jpg
python infer.py --checkpoint checkpoints/best_model_e5_bs16.pt --image-dir ruta/a/carpeta
```
Imprime la clase predicha, la confianza y las probabilidades por clase.

### `app.py`: interfaz local con Gradio
Interfaz web para subir una imagen y ver la clase predicha, la categoría de riesgo (Maligno / Premaligno / Benigno) y las probabilidades por clase, con avisos de uso. Por defecto carga `checkpoints/best_model_e5_bs16.pt`.
```bash
python app.py                                  # http://localhost:7860
python app.py --checkpoint checkpoints/<otro>.pt --port 7860
```

### `export_onnx.py` y `web/`: demo en el navegador
`export_onnx.py` exporta el modelo E5a a `skincancer.onnx` (opset 17), verifica que la salida coincida con PyTorch (diferencia máxima ≈ 5e-6) e imprime las probabilidades de referencia de una imagen. La carpeta `web/` es un sitio estático (`index.html`, `assets/app.js`, `assets/style.css`) que descarga el `.onnx` y lo ejecuta con `onnxruntime-web`: **la imagen no sale del navegador del usuario**. El preprocesamiento de `app.js` reproduce el de `get_transforms(train=False)` (redimensionado bilineal con antialias a 224×224 y normalización ImageNet).

```bash
python export_onnx.py /ruta/a/imagen.jpg      # genera skincancer.onnx
# colocar skincancer.onnx en web/model/ y servir web/ con cualquier servidor estático:
cd web && python -m http.server 8000
```

## 7. Entrenamiento en el clúster HPC (SLURM)

Los entrenamientos se ejecutaron en el clúster **Océano (PUCV)**, partición `GPU`, con trabajos SLURM. Ejemplo (`sbatch/train_job.sbatch`):

```bash
#!/bin/bash
#SBATCH --job-name=lesiones_finetune
#SBATCH --partition=GPU
#SBATCH --qos=normal-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate lesiones
python train.py --data-dir /work/<usuario>/dataset --epochs 30 --batch-size 32 \
  --num-workers 8 --lr 1e-5 --no-freeze-backbone \
  --checkpoint-path checkpoints/best_model_finetune.pt
```

```bash
sbatch sbatch/train_job.sbatch
squeue -u $USER
```

**Importante:** lanza un trabajo a la vez en la misma GPU; al ejecutar dos en paralelo se produjo un error de memoria (`CUDA out of memory`). El orden de ejecución de los experimentos y los archivos que deja cada trabajo están en [`docs/GUIA_EJECUCION_HPC.md`](docs/GUIA_EJECUCION_HPC.md).

Scripts incluidos en `sbatch/`: `train_job`, `train_finetune`, `train_ablation_convnext`, `train_ablation_swin`, `train_e2_earlystop`, `train_e3_scheduler`, `train_e4_dropout`, `train_e5_bs16`, `train_e5_bs64`, `train_e6_focal`, `train_e7_combinado`, `train_e5a_repeticiones` y `export_onnx`. Adapta las rutas (`/work/<usuario>/...`) y las líneas de partición y entorno a tu clúster.

## 8. Pesos del modelo

Descarga desde la sección **Releases** de este repositorio:

| Archivo | Descripción | Tamaño |
|---|---|---|
| `best_model_e5_bs16.pt` | Mejor modelo (E5a, 87,16 % en test) | 217 MB |
| `skincancer.onnx` | El mismo modelo exportado a ONNX para el navegador | 216 MB |

Colócalos en `checkpoints/` y `web/model/` respectivamente.

## 9. Limitaciones

- **Desplazamiento de dominio:** el modelo solo se entrenó con imágenes dermatoscópicas; con fotografías clínicas convencionales puede equivocarse. En las pruebas de la interfaz, una fotografía externa de un melanoma fue clasificada como lesión vascular (VASC), error que no ocurre de ese modo en los 678 casos de melanoma del test.
- **Posible fuga de información:** no se verificaron imágenes duplicadas en ISIC 2019, que proviene de varias fuentes; las métricas podrían estar infladas.
- **Comparación de la ablación:** ConvNeXt solo y Swin solo se entrenaron una vez y sin las técnicas de optimización del híbrido; el híbrido optimizado se evaluó en cinco corridas.
- **Fusión:** solo se probó la concatenación simple (no atención cruzada ni gating aprendido).
- **Validación clínica:** el prototipo no fue evaluado por especialistas.
- **Versiones tiny:** se usaron las versiones pequeñas de ambas redes por limitaciones de cómputo.

## 10. Trabajo futuro

Mecanismos de fusión más expresivos (atención cruzada, gating), comparación pareja de los modelos individuales con la misma configuración y repeticiones, verificación de duplicados, mejor desempeño en clases minoritarias y SCC, y datasets que incluyan fotografías clínicas no dermatoscópicas.

## 11. Referencias principales

- Liu et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV.
- Liu et al. (2022). *A ConvNet for the 2020s* (ConvNeXt). CVPR.
- Naeem et al. (2025). *MedFusionNet: A hybrid ConvNeXt and Vision Transformer network for skin lesion classification.* Scientific Reports, 15, 31816.
- Lin et al. (2017). *Focal Loss for Dense Object Detection.*
- Loshchilov y Hutter (2019). *Decoupled Weight Decay Regularization* (AdamW).
- Tschandl, Rosendahl y Kittler (2018). *The HAM10000 dataset.* Scientific Data.
- Codella et al. (2018). *Skin lesion analysis toward melanoma detection (ISBI 2017 challenge).* arXiv:1710.05006.
- Wightman (2019). *PyTorch Image Models (timm).* · Paszke et al. (2019). *PyTorch.* NeurIPS.

El informe completo del proyecto de título contiene la bibliografía completa.
