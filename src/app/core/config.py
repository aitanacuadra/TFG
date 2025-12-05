import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Definimos valores por defecto, pero leerá tu .env automáticamente
    API_KEY: str = os.getenv("API_KEY", "dev-key")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./runs.db")
    BASE_DATASET_URL: str = os.getenv("BASE_DATASET_URL", "https://example.org/dataset")
    BASE_DOWNLOAD_URL: str = os.getenv("BASE_DOWNLOAD_URL", "https://example.org/datasets")

    class Config:
        env_file = ".env"  # Busca el archivo .env en la raíz

settings = Settings()