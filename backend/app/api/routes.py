from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.models.predictor import predictor
from app.schemas.prediction import PredictionResponse

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    if file.filename is None or not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a valid image file (jpg, jpeg, png, or webp).",
        )

    try:
        image_bytes = await file.read()
        result = predictor.predict_from_bytes(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return PredictionResponse(**result)
