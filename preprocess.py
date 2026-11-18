"""
Preprocesamiento de dataset de lesiones cutaneas (ISIC 2019, 8 clases).

Lee un dataset con estructura por clases (AK/BCC/BKL/DF/MEL/NV/SCC/VASC),
verifica las imagenes, genera un split estratificado train/val/test y
construye Dataset/DataLoader de PyTorch con augmentation solo en train.

Uso:
    python preprocess.py --data-dir dataset_sample --batch-size 8
"""

import argparse
from pathlib import Path
from collections import Counter

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def scan_dataset(data_dir: Path):
    """Recorre las carpetas por clase y valida cada imagen (abre + RGB)."""
    samples = []
    skipped = []

    for class_name in CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            print(f"[WARN] Carpeta de clase no encontrada, se omite: {class_dir}")
            continue

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                skipped.append((img_path, str(e)))
                continue
            samples.append((str(img_path), class_name))

    if skipped:
        print(f"[WARN] {len(skipped)} imagenes invalidas omitidas:")
        for path, err in skipped[:10]:
            print(f"  - {path}: {err}")

    if not samples:
        raise RuntimeError(f"No se encontraron imagenes validas en {data_dir}")

    return samples


def stratified_split(samples, val_size, test_size, seed):
    """Split estratificado por clase en train/val/test."""
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    val_relative_size = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_relative_size,
        stratify=train_val_labels,
        random_state=seed,
    )

    return {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
    }


def print_class_distribution(splits):
    """Imprime la distribucion de clases por split."""
    print("\nDistribucion de clases por split:")
    header = f"{'Clase':<8}" + "".join(f"{name:>10}" for name in splits)
    print(header)

    for class_name in CLASSES:
        row = f"{class_name:<8}"
        for split_name in splits:
            count = sum(1 for _, label in splits[split_name] if label == class_name)
            row += f"{count:>10}"
        print(row)

    total_row = f"{'TOTAL':<8}"
    for split_name in splits:
        total_row += f"{len(splits[split_name]):>10}"
    print(total_row)


def get_transforms(train: bool):
    """Transforms: resize 224x224 + normalizacion ImageNet; augmentation solo en train."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ISICDataset(Dataset):
    """Dataset de lesiones cutaneas a partir de una lista de (ruta, clase)."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_name = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[class_name]
        return image, label


def build_dataloaders(splits, batch_size, num_workers=0):
    """Construye un DataLoader por split (augmentation solo en train)."""
    datasets = {
        "train": ISICDataset(splits["train"], get_transforms(train=True)),
        "val": ISICDataset(splits["val"], get_transforms(train=False)),
        "test": ISICDataset(splits["test"], get_transforms(train=False)),
    }

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocesamiento de dataset de lesiones cutaneas")
    parser.add_argument("--data-dir", type=str, required=True,
                         help="Ruta al dataset con carpetas por clase (AK/BCC/.../VASC)")
    parser.add_argument("--val-size", type=float, default=0.15, help="Proporcion para validacion")
    parser.add_argument("--test-size", type=float, default=0.15, help="Proporcion para test")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (chico para GPU local)")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers para DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para el split")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de dataset: {data_dir}")

    print(f"Escaneando dataset en: {data_dir}")
    samples = scan_dataset(data_dir)
    print(f"Total de imagenes validas: {len(samples)}")
    print(f"Distribucion global: {dict(Counter(label for _, label in samples))}")

    splits = stratified_split(samples, args.val_size, args.test_size, args.seed)
    print_class_distribution(splits)

    loaders = build_dataloaders(splits, args.batch_size, args.num_workers)

    print("\nVerificando un batch por split...")
    for split_name, loader in loaders.items():
        images, labels = next(iter(loader))
        print(f"  {split_name}: batch images={tuple(images.shape)}, labels={tuple(labels.shape)}")


if __name__ == "__main__":
    main()
