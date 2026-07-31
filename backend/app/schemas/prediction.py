from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image payload")


class TopPrediction(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
