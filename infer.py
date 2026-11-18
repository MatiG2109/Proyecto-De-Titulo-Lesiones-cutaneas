"""
Inferencia con el modelo hibrido ConvNeXt + Swin para clasificacion
de lesiones cutaneas (ISIC 2019, 8 clases).

Uso (una imagen):
    python infer.py --checkpoint checkpoints/best_model_finetune.pt --image ruta/a/imagen.jpg

Uso (carpeta de imagenes):
    python infer.py --checkpoint checkpoints/best_model_finetune.pt --image-dir ruta/a/carpeta
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from preprocess import CLASSES, get_transforms
from hybrid_model import HybridConvNeXtSwin


def load_model(checkpoint_path, device):
    model = HybridConvNeXtSwin(num_classes=len(CLASSES), pretrained=False, freeze_backbone=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_image(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = torch.argmax(probs).item()
    pred_class = CLASSES[pred_idx]
    return pred_class, probs


def print_prediction(image_path, pred_class, probs):
    print(f"\nImagen: {image_path}")
    print(f"Prediccion: {pred_class} (confianza: {probs[CLASSES.index(pred_class)]:.4f})")
    print("Probabilidades por clase:")
    for cls, p in sorted(zip(CLASSES, probs.tolist()), key=lambda x: -x[1]):
        print(f"  {cls:<6}: {p:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia de lesiones cutaneas")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al checkpoint (.pt)")
    parser.add_argument("--image", type=str, help="Ruta a una imagen individual")
    parser.add_argument("--image-dir", type=str, help="Ruta a una carpeta con varias imagenes")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image and not args.image_dir:
        raise ValueError("Debes especificar --image o --image-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    transform = get_transforms(train=False)

    if args.image:
        pred_class, probs = predict_image(model, args.image, transform, device)
        print_prediction(args.image, pred_class, probs)

    if args.image_dir:
        image_dir = Path(args.image_dir)
        valid_ext = {".jpg", ".jpeg", ".png"}
        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in valid_ext)

        if not image_paths:
            print(f"No se encontraron imagenes validas en {image_dir}")
            return

        for img_path in image_paths:
            pred_class, probs = predict_image(model, img_path, transform, device)
            print_prediction(img_path, pred_class, probs)


if __name__ == "__main__":
    main()
