from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List

class Settings(BaseSettings):
    app_name: str = "QuantumAI Medical Platform"
    environment: str = "development"

    # Security
    secret_key: str = Field("change-me", validation_alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(60 * 8, validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field("HS256", validation_alias="ALGORITHM")

    # Default admin seed
    admin_email: str = Field("admin@quantumai.local", validation_alias="ADMIN_EMAIL")
    admin_password: str = Field("admin123", validation_alias="ADMIN_PASSWORD")

    # Database
    # Default to SQLite locally; Docker overrides via .env
    database_url: str = Field("sqlite:///./quantumai.db", validation_alias="DATABASE_URL")

    # Model paths
    brain_model_path: str = Field("models/saved/brain_effresnet_vit_classifier.h5", validation_alias="BRAIN_MODEL_PATH")
    lung_model_path: str = Field("models/saved/lung_effresnet_vit_classifier.h5", validation_alias="LUNG_MODEL_PATH")

    # CORS
    cors_origins: str = Field("*", validation_alias="CORS_ORIGINS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
