from fastapi import FastAPI
# FÍJATE AQUÍ: Debe decir 'from app.core...' NO 'from src.app.core...'
from app.core.config import settings 
from app.db.session import init_db
from app.api.v1 import endpoints

app = FastAPI(title="API TFG")

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(endpoints.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)