from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    API_KEY: str = "dev-key"
    DATABASE_URL: str = "sqlite:///./runs.db"
    BASE_DATASET_URL: str = "https://example.org/dataset"
    BASE_DOWNLOAD_URL: str = "https://example.org/datasets"
    
    
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3-flash-preview"
    
  
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:1b"
    
    
    EMBEDDINGS_MODEL: str = "nomic-embed-text"
    FAISS_INDEX_PATH: str = "faiss_index"
    RAG_DOC_PATH: str = "src/data/docs/dcat_ap.html"

    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()