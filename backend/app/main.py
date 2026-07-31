from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as prediction_router
from app.core.config import settings

app = FastAPI(title="Crop Leaf Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
