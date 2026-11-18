"""
Modelos de ablacion (un solo backbone) para comparar contra HybridConvNeXtSwin.

Permite entrenar y evaluar ConvNeXt-tiny solo o Swin Transformer-tiny solo,
usando exactamente el mismo pipeline de datos y entrenamiento que el modelo
hibrido, para aislar el efecto de combinar ambas arquitecturas.

Uso (test rapido):
    python ablation_model.py --data-dir dataset_sample --batch-size 4 --backbone convnext
"""

import argparse

import torch
import torch.nn as nn
import timm

from preprocess import CLASSES, build_dataloaders, scan_dataset, stratified_split

CONVNEXT_MODEL = "convnext_tiny"
SWIN_MODEL = "swin_tiny_patch4_window7_224"
NUM_CLASSES = len(CLASSES)


class SingleBackboneModel(nn.Module):
    """Clasificador con un unico backbone (ConvNeXt o Swin), sin fusion."""

    def __init__(self, backbone_name, num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def build_ablation_model(backbone, pretrained=True, freeze_backbone=False):
    if backbone == "convnext":
        return SingleBackboneModel(CONVNEXT_MODEL, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif backbone == "swin":
        return SingleBackboneModel(SWIN_MODEL, pretrained=pretrained, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Backbone desconocido: {backbone}. Usa 'convnext' o 'swin'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Test rapido de modelos de ablacion")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, choices=["convnext", "swin"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from pathlib import Path
    samples = scan_dataset(Path(args.data_dir))
    splits = stratified_split(samples, args.val_size, args.test_size, args.seed)
    loaders = build_dataloaders(splits, args.batch_size, args.num_workers)

    model = build_ablation_model(args.backbone).to(device)
    print(f"Modelo de ablacion ({args.backbone}) creado. Verificando un batch...")

    images, labels = next(iter(loaders["train"]))
    images = images.to(device)
    output = model(images)
    print(f"Input: {tuple(images.shape)} -> Output: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
