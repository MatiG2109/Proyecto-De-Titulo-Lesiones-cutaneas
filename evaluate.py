"""
Genera la matriz de confusion (como imagen PNG) del modelo entrenado sobre
el conjunto de test, a partir de un checkpoint ya guardado. No reentrena nada.

Uso:
    python evaluate.py --checkpoint checkpoints/best_model_finetune.pt --data-dir /work/matias.gallardo/dataset
"""

import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from preprocess import CLASSES, scan_dataset, stratified_split, build_dataloaders
from hybrid_model import HybridConvNeXtSwin


def load_model(checkpoint_path, device):
    model = HybridConvNeXtSwin(num_classes=len(CLASSES), pretrained=False, freeze_backbone=False)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_predictions(model, loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, classes, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Matriz de confusion guardada en: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluar checkpoint y generar matriz de confusion")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--out", type=str, default="figuras/matriz_confusion_finetuning.png")
    parser.add_argument("--title", type=str, default="Matriz de Confusión — Fine-tuning completo")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print("Recreando el mismo split (misma seed) para obtener el test set original...")
    samples = scan_dataset(Path(args.data_dir))
    splits = stratified_split(samples, args.val_size, args.test_size, args.seed)
    loaders = build_dataloaders(splits, args.batch_size, args.num_workers)

    print(f"Cargando modelo desde: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    print("Evaluando sobre el conjunto de test...")
    y_true, y_pred = get_predictions(model, loaders["test"], device)

    print("\n=== Reporte de clasificación ===")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    plot_confusion_matrix(y_true, y_pred, CLASSES, args.out, args.title)


if __name__ == "__main__":
    main()
