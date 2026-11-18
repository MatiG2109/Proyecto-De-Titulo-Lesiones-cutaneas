"""
Loop de entrenamiento del modelo hibrido ConvNeXt + Swin Transformer para
clasificacion de lesiones cutaneas (8 clases, ISIC 2019).

Usa Cross-Entropy ponderada (para compensar el desbalance de clases) y
AdamW. Guarda el mejor checkpoint segun val loss y al final reporta
metricas completas (accuracy, precision, recall, F1, matriz de confusion)
sobre el split de test.

Uso (prueba rapida end-to-end):
    python train.py --data-dir dataset_sample --epochs 5 --batch-size 4
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocess import CLASSES, build_dataloaders, scan_dataset, stratified_split
from hybrid_model import HybridConvNeXtSwin, NUM_CLASSES


def compute_class_weights(train_samples):
    """Peso inversamente proporcional a la frecuencia de cada clase (normalizado a media 1)."""
    counts = np.array([sum(1 for _, label in train_samples if label == c) for c in CLASSES], dtype=np.float64)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, device):
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
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def print_test_report(all_labels, all_preds):
    print("\n=== Metricas en test ===")
    print(f"Accuracy global: {accuracy_score(all_labels, all_preds):.4f}")

    print("\nReporte por clase (precision / recall / F1):")
    print(classification_report(
        all_labels, all_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASSES,
        zero_division=0,
    ))

    print("Matriz de confusion (filas=real, columnas=prediccion):")
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    header = "        " + "".join(f"{c:>6}" for c in CLASSES)
    print(header)
    for class_name, row in zip(CLASSES, cm):
        print(f"{class_name:<8}" + "".join(f"{v:>6}" for v in row))


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo hibrido ConvNeXt+Swin")
    parser.add_argument("--data-dir", type=str, required=True,
                         help="Ruta al dataset con carpetas por clase (ej. dataset_sample)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (chico para GPU local 6GB)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate para AdamW")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                         help="Congela ConvNeXt y Swin, entrena solo el clasificador")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best_model.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    print(f"Escaneando dataset en: {data_dir}")
    samples = scan_dataset(data_dir)
    splits = stratified_split(samples, args.val_size, args.test_size, args.seed)
    loaders = build_dataloaders(splits, args.batch_size, args.num_workers)
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

    class_weights = compute_class_weights(splits["train"]).to(device)
    print(f"Pesos de clase (train): {dict(zip(CLASSES, class_weights.cpu().tolist()))}")

    print(f"Cargando modelo hibrido (freeze_backbone={args.freeze_backbone})...")
    model = HybridConvNeXtSwin(freeze_backbone=args.freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print("\n=== Entrenamiento ===")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device)

        print(f"Epoca {epoch}/{args.epochs} - "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} - "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Nuevo mejor modelo guardado en {checkpoint_path} (val_loss={val_loss:.4f})")

    print(f"\nCargando mejor checkpoint desde {checkpoint_path} para evaluar en test...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, loaders["test"], criterion, device)
    print(f"Test loss={test_loss:.4f} | Test acc={test_acc:.4f}")

    print_test_report(test_labels, test_preds)


if __name__ == "__main__":
    main()
