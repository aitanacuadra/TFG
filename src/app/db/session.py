from sqlmodel import SQLModel, create_engine, Session
from app.core.config import settings

# check_same_thread es necesario solo para SQLite
engine = create_engine(settings.DATABASE_URL, echo=False, connect_args={"check_same_thread": False})

def init_db():
    SQLModel.metadata.create_all(engine)

def get_db():
    with Session(engine) as session:
        yield session