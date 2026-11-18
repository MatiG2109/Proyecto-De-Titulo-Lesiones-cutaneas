"""
Arquitectura hibrida ConvNeXt + Swin Transformer para clasificacion de
lesiones cutaneas (8 clases, ISIC 2019).

Cada rama actua como extractor de caracteristicas via Transfer Learning
(timm, preentrenados en ImageNet). Las caracteristicas se fusionan por
concatenacion simple y se pasan a una capa clasificadora final.

Uso (test rapido):
    python hybrid_model.py --data-dir dataset_sample --batch-size 4
"""

import argparse

import torch
import torch.nn as nn
import timm

from preprocess import CLASSES, build_dataloaders, scan_dataset, stratified_split

CONVNEXT_MODEL = "convnext_tiny"
SWIN_MODEL = "swin_tiny_patch4_window7_224"
NUM_CLASSES = len(CLASSES)


class HybridConvNeXtSwin(nn.Module):
    """ConvNeXt + Swin Transformer en paralelo, fusion por concatenacion."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True, dropout_rate=0.3):
        super().__init__()

        self.convnext = timm.create_model(CONVNEXT_MODEL, pretrained=pretrained, num_classes=0)
        self.swin = timm.create_model(SWIN_MODEL, pretrained=pretrained, num_classes=0)

        if freeze_backbone:
            for param in self.convnext.parameters():
                param.requires_grad = False
            for param in self.swin.parameters():
                param.requires_grad = False

        fused_dim = self.convnext.num_features + self.swin.num_features

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        convnext_features = self.convnext(x)
        swin_features = self.swin(x)
        fused = torch.cat([convnext_features, swin_features], dim=1)
        return self.classifier(fused)


def parse_args():
    parser = argparse.ArgumentParser(description="Test rapido del modelo hibrido ConvNeXt+Swin")
    parser.add_argument("--data-dir", type=str, default=None,
                         help="Ruta a dataset con carpetas por clase (ej. dataset_sample). Si se omite, usa batch dummy.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (chico para GPU local 6GB)")
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                         help="Congela los pesos de ambas ramas (extraccion de caracteristicas pura)")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    return parser.parse_args()


def get_real_batch(data_dir, batch_size):
    """Reusa preprocess.py para armar un batch real de train desde data_dir."""
    samples = scan_dataset(data_dir)
    splits = stratified_split(samples, val_size=0.15, test_size=0.15, seed=42)
    loaders = build_dataloaders(splits, batch_size)
    images, labels = next(iter(loaders["train"]))
    return images, labels


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.data_dir:
        from pathlib import Path
        print(f"Cargando batch real desde: {args.data_dir}")
        images, labels = get_real_batch(Path(args.data_dir), args.batch_size)
        print(f"Batch real cargado: images={tuple(images.shape)}, labels={tuple(labels.shape)}")
    else:
        print("Sin --data-dir: usando batch dummy")
        images = torch.randn(args.batch_size, 3, 224, 224)
        labels = None

    print(f"Cargando modelo hibrido ({CONVNEXT_MODEL} + {SWIN_MODEL})...")
    model = HybridConvNeXtSwin(freeze_backbone=args.freeze_backbone).to(device)
    model.eval()

    images = images.to(device)

    with torch.no_grad():
        output = model(images)

    print(f"Output shape: {tuple(output.shape)}")
    expected_shape = (images.shape[0], NUM_CLASSES)
    assert output.shape == expected_shape, f"Shape esperado {expected_shape}, obtenido {tuple(output.shape)}"
    print(f"OK: output tiene shape {expected_shape} como se esperaba.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros totales: {total_params:,} | entrenables: {trainable_params:,}")


if __name__ == "__main__":
    main()
