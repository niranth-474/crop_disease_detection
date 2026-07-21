import os
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import get_dataloaders
from models.model import Disease


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, class_names = get_dataloaders()

    model_path = PROJECT_ROOT / "saved_models" / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = Disease(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
    )

    print("=" * 80)
    print("Classification Report")
    print("=" * 80)
    print(report)

    report_path = PROJECT_ROOT / "saved_models" / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(15, 15))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=90, ax=ax, colorbar=False)

    plt.title("Confusion Matrix")
    plt.tight_layout()
    output_path = PROJECT_ROOT / "saved_models" / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved classification report to: {report_path}")
    print(f"Saved confusion matrix to: {output_path}")


if __name__ == "__main__":
    main()