from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field

class Run(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    # Corrección A: Uso correcto y moderno de zonas horarias UTC
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), 
        nullable=False
    )
    endpoint: str
    filename: str
    content_type: str
    size_bytes: int
    provider: str = "ollama"
    model: str = "gemma3:1b"
    status: str = "started"
    error: Optional[str] = None 

class RunMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id")
    
    metadata_json: str
    evaluation_json: Optional[str] = None