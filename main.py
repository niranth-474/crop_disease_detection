import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import get_dataloaders
from models.model import Disease
from engine.train import train_one_epoch, validate_one_epoch


# Hyperparameters
NUM_EPOCHS = 30
LEARNING_RATE = 0.001


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device : {device}")


# Data
train_loader, val_loader, test_loader, classes = get_dataloaders()


# Model
model = Disease(num_classes=len(classes))
model = model.to(device)


# Loss Function
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=5e-4
)

# Scheduler - halves LR when val loss plateaus for 3 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

best_val_acc = 0.0
start_epoch = 0

# Training Loop
for epoch in range(start_epoch, NUM_EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_acc = validate_one_epoch(
        model,
        val_loader,
        criterion,
        device
    )

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # Save checkpoint every epoch - lets you resume without retraining from scratch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'best_val_acc': best_val_acc,
    }, "saved_models/last_checkpoint.pth")

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "saved_models/best_model.pth"
        )

        print("Best model saved!")

    print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
    print("-" * 40)
    print(f"Train Loss : {train_loss:.4f}    Train Acc  : {train_acc:.2f}%")
    print(f"Val Loss   : {val_loss:.4f}    Val Acc    : {val_acc:.2f}%")
    print(f"LR         : {current_lr:.6f}")