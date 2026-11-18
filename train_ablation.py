"""
Entrenamiento de modelos de ablacion (un solo backbone) para comparar
contra HybridConvNeXtSwin. Reutiliza el mismo pipeline de datos, pesos de
clase, optimizador y metricas que train.py, cambiando solo la arquitectura.

Uso:
    python train_ablation.py --data-dir /work/matias.gallardo/dataset \
        --backbone convnext --epochs 30 --batch-size 32 --lr 1e-5 \
        --checkpoint-path checkpoints/best_model_convnext_only.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from preprocess import CLASSES, build_dataloaders, scan_dataset, stratified_split
from ablation_model import build_ablation_model


def compute_class_weights(splits, device):
    labels = [CLASSES.index(lbl) for _, lbl in splits["train"]]
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASSES)), y=labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_labels, all_preds


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo de ablacion")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, choices=["convnext", "swin"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone de ablacion: {args.backbone}")

    data_dir = Path(args.data_dir)
    print(f"Escaneando dataset en: {data_dir}")
    samples = scan_dataset(data_dir)
    splits = stratified_split(samples, args.val_size, args.test_size, args.seed)
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

    loaders = build_dataloaders(splits, args.batch_size, args.num_workers)
    class_weights = compute_class_weights(splits, device)
    print(f"Pesos de clase: {dict(zip(CLASSES, class_weights.cpu().numpy().round(4)))}")

    print(f"Cargando modelo de ablacion ({args.backbone}, fine-tuning completo)...")
    model = build_ablation_model(args.backbone, pretrained=True, freeze_backbone=False).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    print("\n=== Entrenamiento ===")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        print(f"Epoca {epoch}/{args.epochs} - train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"- val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint_path)

    print(f"\nCargando mejor checkpoint desde {args.checkpoint_path} para evaluar en test...")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device, weights_only=True))
    test_loss, test_acc, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
    print(f"Test loss={test_loss:.4f} | Test acc={test_acc:.4f}")

    print("\n=== Reporte de clasificacion ===")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))


if __name__ == "__main__":
    main()
