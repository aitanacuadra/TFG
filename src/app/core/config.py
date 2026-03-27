import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    
    API_KEY: str = os.getenv("API_KEY", "dev-key")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./runs.db")
    BASE_DATASET_URL: str = os.getenv("BASE_DATASET_URL", "https://example.org/dataset")
    BASE_DOWNLOAD_URL: str = os.getenv("BASE_DOWNLOAD_URL", "https://example.org/datasets")
    
    
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    
  
    EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    RAG_DOC_PATH: str = os.getenv("RAG_DOC_PATH", "src/data/docs/dcat_ap.html")

    class Config:
        env_file = ".env"  

settings = Settings()