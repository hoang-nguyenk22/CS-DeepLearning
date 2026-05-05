import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import random
import numpy as np

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(data_dir='./data', batch_size=32, num_workers=4, augment_level='light', seed=42):
    """
    Loads torchvision FGVC-Aircraft dataset and returns strictly separate train, val, test loaders.
    """
    print(f"Loading FGVC-Aircraft from {data_dir} (Augmentation: {augment_level}, Seed: {seed})")
    
    # 1. Separate generators for each loader — prevent shared random state bug
    g_train = torch.Generator(); g_train.manual_seed(seed)
    g_val   = torch.Generator(); g_val.manual_seed(seed)
    g_test  = torch.Generator(); g_test.manual_seed(seed)
    
    # Define transformations dynamically based on augment_level
    if augment_level == 'strong':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # Light augmentation
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Strict Splits to prevent Data Leakage
    train_dataset = datasets.FGVCAircraft(
        root=data_dir, split='train', annotation_level='variant', 
        download=True, transform=train_transform
    )
    
    val_dataset = datasets.FGVCAircraft(
        root=data_dir, split='val', annotation_level='variant', 
        download=True, transform=val_test_transform
    )
    
    test_dataset = datasets.FGVCAircraft(
        root=data_dir, split='test', annotation_level='variant', 
        download=True, transform=val_test_transform
    )

    # Create DataLoaders
    # Train loader with seeded worker_init_fn for reproducibility
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g_train
    )
    # Val/Test loaders: no shuffle, no augment — seed_worker not needed
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Extract class names
    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names

if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader, class_names = get_dataloaders(num_workers=0)
    print(f"Num classes: {len(class_names)}")
    print(f"Classes: {class_names[:5]}...")
    
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break  
