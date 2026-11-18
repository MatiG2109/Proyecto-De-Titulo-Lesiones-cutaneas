"""
Exporta el modelo híbrido a ONNX para ejecutarlo en el navegador.

Ejecutar en la carpeta de la tesis, con el venv activado:
    pip install onnx onnxruntime
    python export_onnx.py
    python export_onnx.py ruta/a/una_imagen.jpg      (opcional: imprime probabilidades de referencia)

Genera: skincancer.onnx
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F

from preprocess import CLASSES, get_transforms
from hybrid_model import HybridConvNeXtSwin

CKPT = "checkpoints/best_model_e5_bs16.pt"
OUT = "skincancer.onnx"

tf = get_transforms(train=False)
print("\n=== Transformaciones de inferencia (envíame esto) ===")
print(tf)
print("Clases:", CLASSES)

model = HybridConvNeXtSwin(num_classes=len(CLASSES), pretrained=False, freeze_backbone=False)
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
model.eval()

x = torch.randn(1, 3, 224, 224)
print("\nExportando a ONNX (puede tardar un par de minutos)...")
torch.onnx.export(
    model, x, OUT,
    input_names=["input"], output_names=["logits"],
    opset_version=17, do_constant_folding=True,
)
print("Archivo generado:", OUT)

# Verificación: ONNX vs PyTorch con la misma entrada
import onnxruntime as ort

sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
with torch.no_grad():
    ref = model(x).numpy()
out = sess.run(None, {"input": x.numpy()})[0]
print("\nDiferencia máxima PyTorch vs ONNX (debe ser < 1e-3):", float(np.abs(ref - out).max()))

# Probabilidades de referencia para una imagen real (para comparar con el navegador)
if len(sys.argv) > 1:
    from PIL import Image

    img = Image.open(sys.argv[1]).convert("RGB")
    t = tf(img).unsqueeze(0)
    with torch.no_grad():
        p = F.softmax(model(t), dim=1).squeeze(0).tolist()
    print("\nProbabilidades de referencia (PyTorch):")
    for c, v in sorted(zip(CLASSES, p), key=lambda z: -z[1]):
        print(f"  {c:5s} {v * 100:5.1f}%")
