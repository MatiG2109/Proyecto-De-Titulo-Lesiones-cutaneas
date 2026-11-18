"""
Version 2 del loop de entrenamiento del modelo hibrido ConvNeXt + Swin
Transformer, con tecnicas adicionales de regularizacion y optimizacion
solicitadas para las nuevas pruebas del seminario de titulo:

  - Dropout ajustable          --dropout-rate
  - Early stopping             --early-stopping-patience
  - LR scheduler               --scheduler {none,cosine,plateau,onecycle}
  - Label smoothing            --label-smoothing
  - Focal loss (vs CE ponderada) --use-focal-loss / --focal-gamma
  - Mixed precision (AMP)      --amp
  - Gradient clipping          --grad-clip
  - Weight decay               --weight-decay

Con todos los flags en su valor por defecto, este script entrena de forma
IDENTICA a train.py (mismo dropout=0.3, mismo weight_decay=0.01 de AdamW,
sin scheduler, sin early stopping, sin focal loss).

train.py NO se modifica: este es un script nuevo e independiente.

Uso (ejemplos, ver tabla de experimentos en nuevas_pruebas_regularizacion.md):
    # Early stopping
    python train_v2.py --data-dir dataset_sample --epochs 30 --no-freeze-backbone \
        --lr 1e-5 --early-stopping-patience 5 \
        --checkpoint-path checkpoints/best_model_e2_earlystop.pt

    # Scheduler coseno
    python train_v2.py --data-dir dataset_sample --epochs 30 --no-freeze-backbone \
        --lr 1e-5 --scheduler cosine \
        --checkpoint-path checkpoints/best_model_e3_cosine.pt

    # Dropout aumentado
    python train_v2.py --data-dir dataset_sample --epochs 30 --no-freeze-backbone \
        --lr 1e-5 --dropout-rate 0.5 \
        --checkpoint-path checkpoints/best_model_e4_dropout.pt

    # Batch size distinto
    python train_v2.py --data-dir dataset_sample --epochs 30 --no-freeze-backbone \
        --lr 1e-5 --batch-size 16 \
        --checkpoint-path checkpoints/best_model_e5_bs16.pt

    # Focal loss
    python train_v2.py --data-dir dataset_sample --epochs 30 --no-freeze-backbone \
        --lr 1e-5 --use-focal-loss --focal-gamma 2.0 \
        --checkpoint-path checkpoints/best_model_e6_focal.pt
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from preprocess import CLASSES, build_dataloaders, scan_dataset, stratified_split
from hybrid_model import HybridConvNeXtSwin, NUM_CLASSES


# ---------------------------------------------------------------------------
# Perdidas
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017), alternativa a CE ponderada para desbalance
    de clases. Da menos peso relativo a ejemplos ya bien clasificados (pt alto)
    y se concentra en los dificiles."""

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_class_weights(train_samples):
    """Peso inversamente proporcional a la frecuencia de cada clase (normalizado a media 1)."""
    counts = np.array([sum(1 for _, label in train_samples if label == c) for c in CLASSES], dtype=np.float64)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Entrenamiento / evaluacion
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=None, onecycle_scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if onecycle_scheduler is not None:
            onecycle_scheduler.step()

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

    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, correct / total, f1_macro, all_preds, all_labels


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def set_training_seed(seed, deterministic=False):
    """Fija las fuentes de aleatoriedad del ENTRENAMIENTO (inicializacion del
    clasificador, dropout, orden de los batches y data augmentation).

    Es independiente de --seed, que solo controla la particion estratificada
    train/val/test. Asi el conjunto de test es siempre el mismo y lo unico que
    cambia entre repeticiones es la aleatoriedad propia del entrenamiento.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Mas reproducible en GPU, a costa de algo de velocidad.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento v2 del modelo hibrido ConvNeXt+Swin (nuevas pruebas)")
    parser.add_argument("--data-dir", type=str, required=True,
                         help="Ruta al dataset con carpetas por clase (ej. dataset_sample)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (chico para GPU local 6GB)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate para AdamW")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42,
                         help="Semilla de la particion estratificada train/val/test (no afecta el entrenamiento)")
    parser.add_argument("--train-seed", type=int, default=None,
                         help="Semilla del entrenamiento (init del clasificador, dropout, orden de batches y "
                              "augmentation). Por defecto None = sin fijar, igual que en E2-E7.")
    parser.add_argument("--deterministic", action="store_true", default=False,
                         help="Activa cudnn.deterministic (mas reproducible en GPU, algo mas lento)")
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                         help="Congela ConvNeXt y Swin, entrena solo el clasificador")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best_model_v2.pt")

    # --- Nuevas tecnicas ---
    parser.add_argument("--dropout-rate", type=float, default=0.3,
                         help="Tasa de dropout en el clasificador (default: igual que train.py)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                         help="Weight decay para AdamW (default: igual al valor por defecto de AdamW)")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                         help="Detiene el entrenamiento si val_f1_macro no mejora tras N epocas (default: desactivado)")
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine", "plateau", "onecycle"], default="none",
                         help="Learning rate scheduler (default: ninguno, igual que train.py)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                         help="Label smoothing para CrossEntropyLoss (default: 0, desactivado)")
    parser.add_argument("--use-focal-loss", action="store_true", default=False,
                         help="Usa Focal Loss en vez de CE ponderada (mantiene los pesos por clase)")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma de Focal Loss")
    parser.add_argument("--amp", action="store_true", default=False,
                         help="Mixed precision (float16) durante el entrenamiento, solo en GPU")
    parser.add_argument("--grad-clip", type=float, default=None,
                         help="Norma maxima para gradient clipping (default: desactivado)")
    parser.add_argument("--history-csv", type=str, default=None,
                         help="Ruta CSV para historial por epoca (default: logs/<checkpoint>_history.csv)")

    return parser.parse_args()


def build_scheduler(args, optimizer, steps_per_epoch):
    """Devuelve (scheduler_por_epoca, scheduler_por_batch). Solo uno de los dos
    tendra valor distinto de None, segun el tipo elegido."""
    if args.scheduler == "none":
        return None, None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs), None
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3), None
    if args.scheduler == "onecycle":
        onecycle = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs,
        )
        return None, onecycle
    raise ValueError(f"Scheduler desconocido: {args.scheduler}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.train_seed is not None:
        set_training_seed(args.train_seed, deterministic=args.deterministic)
        print(f"Semilla de entrenamiento fijada: {args.train_seed} (deterministic={args.deterministic})")
    else:
        print("Semilla de entrenamiento: NO fijada (comportamiento aleatorio, como en E2-E7)")
    print(f"Semilla del split estratificado: {args.seed}")

    if args.amp and device.type != "cuda":
        print("Aviso: --amp solicitado pero no hay GPU disponible, se ignora.")
        args.amp = False

    if args.use_focal_loss and args.label_smoothing > 0:
        print("Aviso: --use-focal-loss activo, se ignora --label-smoothing (no aplica a Focal Loss).")

    data_dir = Path(args.data_dir)
    print(f"Escaneando dataset en: {data_dir}")
    samples = scan_dataset(data_dir)
    splits = stratified_split(samples, args.val_size, args.test_size, args.seed)
    loaders = build_dataloaders(splits, args.batch_size, args.num_workers)
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

    class_weights = compute_class_weights(splits["train"]).to(device)
    print(f"Pesos de clase (train): {dict(zip(CLASSES, class_weights.cpu().tolist()))}")

    print(f"Cargando modelo hibrido (freeze_backbone={args.freeze_backbone}, dropout_rate={args.dropout_rate})...")
    model = HybridConvNeXtSwin(freeze_backbone=args.freeze_backbone, dropout_rate=args.dropout_rate).to(device)

    if args.use_focal_loss:
        criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma)
        print(f"Loss: Focal Loss (gamma={args.focal_gamma}, con pesos de clase)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        print(f"Loss: CrossEntropy ponderada (label_smoothing={args.label_smoothing})")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    epoch_scheduler, onecycle_scheduler = build_scheduler(args, optimizer, len(loaders["train"]))
    print(f"Scheduler: {args.scheduler}")

    scaler = torch.amp.GradScaler(device.type) if args.amp else None
    if args.amp:
        print("AMP (mixed precision) activado.")
    if args.grad_clip is not None:
        print(f"Gradient clipping activado (max_norm={args.grad_clip}).")
    if args.early_stopping_patience is not None:
        print(f"Early stopping activado (patience={args.early_stopping_patience}, monitorea val_f1_macro).")

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history_csv = Path(args.history_csv) if args.history_csv else Path("logs") / f"{checkpoint_path.stem}_history.csv"
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(history_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro", "lr"])
    print(f"Historial por epoca se guardara en: {history_csv}")

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    print("\n=== Entrenamiento ===")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, loaders["train"], criterion, optimizer, device,
            scaler=scaler, grad_clip=args.grad_clip, onecycle_scheduler=onecycle_scheduler,
        )
        val_loss, val_acc, val_f1_macro, _, _ = evaluate(model, loaders["val"], criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoca {epoch}/{args.epochs} - "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} - "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1_macro={val_f1_macro:.4f} - "
              f"lr={current_lr:.2e}")

        with open(history_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc, val_f1_macro, current_lr])

        # Checkpoint: usa val_f1_macro como criterio en TODOS los casos.
        # (En pruebas con este dataset desbalanceado, val_loss selecciona de
        # forma consistente checkpoints peores que val_f1_macro: el loss deja
        # de bajar mientras el F1 macro sigue mejorando en epocas posteriores.)
        improved = val_f1_macro > best_val_f1

        if improved:
            best_val_f1 = val_f1_macro
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Nuevo mejor modelo guardado en {checkpoint_path} "
                  f"(val_loss={val_loss:.4f}, val_f1_macro={val_f1_macro:.4f})")

        # Scheduler por epoca
        if epoch_scheduler is not None:
            if args.scheduler == "plateau":
                epoch_scheduler.step(val_loss)
            else:
                epoch_scheduler.step()

        # Early stopping (monitorea val_f1_macro, misma señal que el checkpoint)
        if args.early_stopping_patience is not None:
            if improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"  -> Sin mejora en val_f1_macro hace {epochs_without_improvement} epoca(s) "
                      f"(mejor hasta ahora: {best_val_f1:.4f})")
                if epochs_without_improvement >= args.early_stopping_patience:
                    print(f"Early stopping activado en la epoca {epoch}.")
                    break

    print(f"\nCargando mejor checkpoint desde {checkpoint_path} para evaluar en test...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_loss, test_acc, test_f1_macro, test_preds, test_labels = evaluate(model, loaders["test"], criterion, device)
    print(f"Test loss={test_loss:.4f} | Test acc={test_acc:.4f} | Test F1 macro={test_f1_macro:.4f}")

    # Linea parseable (grep RESULTADO_FINAL) para tabular las repeticiones
    f1_weighted = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
    print(f"RESULTADO_FINAL split_seed={args.seed} train_seed={args.train_seed} "
          f"test_acc={test_acc:.4f} test_f1_macro={test_f1_macro:.4f} test_f1_weighted={f1_weighted:.4f}")

    print_test_report(test_labels, test_preds)


if __name__ == "__main__":
    main()
