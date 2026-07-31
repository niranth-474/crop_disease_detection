from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {
        "protected_namespaces": ("settings_",),
        "env_file": ".env",
    }

    app_name: str = "Crop Leaf Disease Detection API"
    model_path: str = ""
    class_names_path: str = ""
    model_url: str = ""
    class_names_url: str = ""
    allowed_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    def model_post_init(self, __context) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.model_path = self.model_path or str(project_root / "saved_models" / "best_model.pth")
        self.class_names_path = self.class_names_path or str(project_root / "PlantVillage" / "class_names.json")
        if isinstance(self.allowed_origins, str):
            self.allowed_origins = [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


settings = Settings()
