from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class Run(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
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
    run_id: int
    metadata_json: str