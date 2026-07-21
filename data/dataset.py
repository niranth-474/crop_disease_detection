import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

IMAGE_SIZE = 256
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42
SIZE = 286


def get_dataloaders():

    # Training Transform
    train_transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    ])

    # Validation/Test Transform
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    ])

    

    
    train_dataset = datasets.ImageFolder(
        root="PlantVillage",
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root="PlantVillage",
        transform=test_transform
    )

    
    train_size = int(TRAIN_RATIO * len(train_dataset))
    val_size = int(VAL_RATIO * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(train_dataset), generator=generator)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(test_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, train_dataset.dataset.classes
