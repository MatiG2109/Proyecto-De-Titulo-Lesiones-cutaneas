"""
Interfaz web (Gradio) para clasificacion de lesiones cutaneas.

Permite a un medico subir una imagen de una lesion cutanea y obtener:
- La clase especifica predicha (8 clases ISIC 2019)
- Una categoria agregada de riesgo (Maligno / Premaligno / Benigno)
- El detalle de probabilidades por clase

Uso:
    python app.py
    (usa por defecto el mejor checkpoint, best_model_e5_bs16.pt; para usar otro:
     python app.py --checkpoint checkpoints/<otro_checkpoint>.pt)
"""

import argparse

import gradio as gr
import torch
import torch.nn.functional as F

from preprocess import CLASSES, get_transforms
from hybrid_model import HybridConvNeXtSwin

# Categoria clinica agregada por clase (ISIC 2019 / dermatologia general).
# MEL: melanoma | BCC: carcinoma basocelular | SCC: carcinoma escamoso -> malignos
# AK: queratosis actinica -> premaligna (lesion precursora)
# NV, BKL, DF, VASC -> benignas
RISK_CATEGORY = {
    "MEL": "Maligno",
    "BCC": "Maligno",
    "SCC": "Maligno",
    "AK": "Premaligno",
    "NV": "Benigno",
    "BKL": "Benigno",
    "DF": "Benigno",
    "VASC": "Benigno",
}

CLASS_NAMES_ES = {
    "AK": "Queratosis actinica",
    "BCC": "Carcinoma basocelular",
    "BKL": "Queratosis benigna",
    "DF": "Dermatofibroma",
    "MEL": "Melanoma",
    "NV": "Nevus melanocitico",
    "SCC": "Carcinoma escamoso",
    "VASC": "Lesion vascular",
}

RISK_COLOR = {
    "Maligno": "#c0392b",
    "Premaligno": "#e67e22",
    "Benigno": "#27ae60",
}

DISCLAIMER = (
    "⚠️ Herramienta de apoyo desarrollada en el marco de un seminario de titulo. "
    "NO reemplaza el diagnostico ni el criterio clinico de un profesional de la salud. "
    "Todo resultado debe ser confirmado mediante evaluacion dermatologica y, "
    "de ser necesario, biopsia."
)

DOMAIN_NOTE = (
    "📷 Esta herramienta fue entrenada exclusivamente con imagenes dermatoscopicas "
    "(tomadas con dermatoscopio: luz polarizada, contacto directo con la piel, aumento "
    "estandarizado). NO fue validada con fotografias clinicas convencionales (ej. tomadas "
    "con camara de celular sin dermatoscopio); su uso con este tipo de imagenes puede "
    "producir resultados no confiables."
)

_model = None
_device = None
_transform = None


def load_model(checkpoint_path, device):
    model = HybridConvNeXtSwin(num_classes=len(CLASSES), pretrained=False, freeze_backbone=False)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(image):
    if image is None:
        return "Sube una imagen para comenzar.", {}, None

    tensor = _transform(image.convert("RGB")).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    pred_idx = max(range(len(probs)), key=lambda i: probs[i])
    pred_class = CLASSES[pred_idx]
    pred_prob = probs[pred_idx]
    risk = RISK_CATEGORY[pred_class]

    resultado_md = (
        f"## Resultado: {CLASS_NAMES_ES[pred_class]} ({pred_class})\n"
        f"### Categoria: <span style='color:{RISK_COLOR[risk]}'>**{risk}**</span>\n"
        f"Confianza del modelo: **{pred_prob*100:.1f}%**"
    )

    prob_dict = {f"{cls} - {CLASS_NAMES_ES[cls]}": p for cls, p in zip(CLASSES, probs)}

    return resultado_md, prob_dict, risk


def build_interface():
    with gr.Blocks(title="Clasificador de Lesiones Cutaneas") as demo:
        gr.Markdown("# 🔬 Clasificador de Lesiones Cutaneas (ISIC 2019)")
        gr.Markdown(
            "Sube una imagen dermatoscopica de una lesion cutanea para obtener "
            "una clasificacion automatizada de apoyo diagnostico."
        )
        gr.Markdown(f"<div style='background:#fdecea;color:#000;padding:10px;border-radius:6px'>{DISCLAIMER}</div>")
        gr.Markdown(f"<div style='background:#eaf2fd;color:#000;padding:10px;border-radius:6px;margin-top:6px'>{DOMAIN_NOTE}</div>")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Imagen de la lesion")
                submit_btn = gr.Button("Analizar imagen", variant="primary")
            with gr.Column():
                result_md = gr.Markdown(label="Resultado")
                prob_output = gr.Label(label="Probabilidades por clase", num_top_classes=8)

        submit_btn.click(fn=predict, inputs=image_input, outputs=[result_md, prob_output, gr.State()])
        image_input.change(fn=predict, inputs=image_input, outputs=[result_md, prob_output, gr.State()])

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Interfaz web de clasificacion de lesiones cutaneas")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_e5_bs16.pt",
                         help="Ruta al checkpoint (.pt). Por defecto usa el mejor modelo de la ronda "
                              "de optimizacion (E5a, batch size 16, 87.16% accuracy en test)")
    parser.add_argument("--port", type=int, default=7860, help="Puerto local para el servidor Gradio")
    return parser.parse_args()


def main():
    global _model, _device, _transform

    args = parse_args()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {_device}")
    print(f"Cargando modelo desde: {args.checkpoint}")

    _model = load_model(args.checkpoint, _device)
    _transform = get_transforms(train=False)

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
