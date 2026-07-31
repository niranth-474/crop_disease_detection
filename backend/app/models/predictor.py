import json
import sys
import urllib.request
from pathlib import Path

import torch
from torch import nn

from app.core.config import settings
from app.utils.image import decode_image_bytes, decode_image_payload, preprocess_image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import Disease


class DiseasePredictor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self._resolve_asset_path(
            settings.model_path,
            settings.model_url,
            "model weights",
            PROJECT_ROOT / "saved_models" / "best_model.pth",
        )
        self.class_names_path = self._resolve_asset_path(
            settings.class_names_path,
            settings.class_names_url,
            "class names",
            PROJECT_ROOT / "PlantVillage" / "class_names.json",
        )
        self.class_names = self._load_class_names()
        self.model = self._load_model()

    def _resolve_asset_path(self, configured_path: str, download_url: str, description: str, default_path: Path) -> Path:
        asset_path = Path(configured_path).expanduser().resolve() if configured_path else default_path.resolve()
        if asset_path.exists():
            return asset_path

        if not download_url:
            raise FileNotFoundError(f"{description} not found: {asset_path}")

        asset_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(download_url, asset_path)
        except Exception as exc:  # pragma: no cover - runtime download path
            raise FileNotFoundError(f"Could not download {description} from {download_url}: {exc}") from exc

        return asset_path

    def _load_class_names(self) -> list[str]:
        if not self.class_names_path.exists():
            raise FileNotFoundError(f"Class names file not found: {self.class_names_path}")
        with self.class_names_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_model(self) -> nn.Module:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")

        model = Disease(num_classes=len(self.class_names))
        state_dict = torch.load(self.model_path, map_location=self.device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_payload: str) -> dict[str, object]:
        image = decode_image_payload(image_payload)
        return self._predict_from_image(image)

    def predict_from_bytes(self, image_bytes: bytes) -> dict[str, object]:
        image = decode_image_bytes(image_bytes)
        return self._predict_from_image(image)

    def _predict_from_image(self, image) -> dict[str, object]:
        tensor = preprocess_image(image).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        top_prob, top_index = torch.topk(probabilities, k=1)
        best_prediction = {
            "class_name": self.class_names[top_index[0].item()],
            "confidence": round(float(top_prob[0].item() * 100), 2),
        }

        return {
            "predicted_class": best_prediction["class_name"],
            "confidence": best_prediction["confidence"],
        }


predictor = DiseasePredictor()
