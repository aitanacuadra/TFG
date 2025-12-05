from fastapi import Header, HTTPException
from app.core.config import settings

# Esta función ahora vive aislada y feliz aquí
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key