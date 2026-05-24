from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- Configuración Base y Seguridad ---
    API_KEY: str = Field(default="dev-key", description="API Key para autenticar requests")
    DATABASE_URL: str = Field(default="sqlite:///./runs.db", description="URL de la BD relacional (Auditoría)")
    BASE_DATASET_URL: str = "https://example.org/dataset"
    BASE_DOWNLOAD_URL: str = "https://example.org/datasets"
    
    # --- Modelos Cloud  ---
    GEMINI_API_KEY: str = Field(default="", description="Token para acceder a Gemini API")
    GEMINI_MODEL: str = Field(default="gemini-3-flash-preview", description="Modelo usado en judge_service")
    
    # --- Modelos Locales (Generador y Embeddings) ---
    OLLAMA_BASE_URL: str = Field(default="http://host.docker.internal:11434", description="URL del servicio Ollama")
    OLLAMA_MODEL: str = Field(default="gemma3:1b", description="Modelo LLM de inferencia local")
    EMBEDDINGS_MODEL: str = Field(default="nomic-embed-text", description="Modelo para vectorización")
    
    # --- Base de Datos Vectorial (Qdrant) ---
    QDRANT_URL: str = Field(default="http://qdrant:6333", description="URL del contenedor Qdrant en Docker")
    QDRANT_COLLECTION: str = Field(default="dcat_metadata", description="Nombre de la colección vectorial")

    # --- RAG y Contexto ---
    RAG_DOC_PATH: str = Field(default="src/data/docs/dcat_ap.html", description="Documento base para RAG")

    # Configuración de Pydantic
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()