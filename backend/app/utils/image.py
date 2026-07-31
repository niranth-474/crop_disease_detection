import base64
import io

from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms


IMAGE_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def decode_image_payload(payload: str) -> Image.Image:
    """Decode a base64 image payload into a PIL image."""
    try:
        image_bytes = base64.b64decode(payload)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError("The uploaded image is not valid base64 data.") from exc

    return decode_image_bytes(image_bytes)


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    """Validate and decode raw image bytes into a PIL image."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    return image


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize and normalize the image to match the trained model input."""
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform(image).unsqueeze(0)
