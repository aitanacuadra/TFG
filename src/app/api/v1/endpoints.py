import json
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Header
from sqlmodel import Session
from pydantic import BaseModel  

from app.db.session import get_db
from app.models.run import Run, RunMetadata
from app.services import file_service, ai_service, metadata_service
from app.core.config import settings
from app.api.deps import verify_api_key
# Importa aquí tu servicio dcat si ya lo moviste a src/app/services/metadata_service.py
# from app.services import metadata_service 

from app.core.config import settings

router = APIRouter()
class PromptIn(BaseModel):
    prompt: str

# Validación simple de API Key


@router.post("/generate")
def generate_text(
    body: PromptIn, 
    api_key: str = Depends(verify_api_key)
):
    """Endpoint simple para chatear con el modelo."""
    # Aquí llamamos a la función nueva que creamos en el paso 1
    response_text = ai_service.get_simple_chat(body.prompt)
    return {"response": response_text}

@router.post("/process")
@router.post("/process")
async def process_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    # ... (código de creación del Run en DB igual que antes) ...
    # Crear run...
    run = Run(endpoint="/process", filename=file.filename, content_type=file.content_type, size_bytes=len(content))
    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        # 1. Analizar archivo (Pandas)
        df, kind = file_service.sniff_dataframe(content, file.content_type or "")
        profile = file_service.dataframe_profile(df)
        sample = file_service.head_as_csv(df)
        
        # 2. IA genera metadatos crudos
        metadata_raw = ai_service.generate_metadata_with_langchain(profile, sample)
        
        # --- AQUÍ ESTÁ EL CAMBIO IMPORTANTE ---
        # Antes tenías: metadata_dcat = metadata_raw
        # AHORA PON ESTO:
        metadata_dcat = metadata_service.build_dcat3_metadata(
            raw_meta=metadata_raw,
            df=df,
            filename=file.filename or "dataset",
            content_type=file.content_type or "application/octet-stream",
            file_size_bytes=len(content)
        )
        # --------------------------------------

        # 3. Guardar en DB
        db.add(RunMetadata(
            run_id=run.id,
            metadata_json=json.dumps(metadata_dcat, ensure_ascii=False)
        ))
        run.status = "completed"
        db.add(run)
        db.commit()

        return {
            "message": "Procesado con éxito",
            "output_filename": f"{file.filename}_meta.json",
            "metadata": metadata_dcat  # Ahora devolverá el DCAT completo
        }

    except Exception as e:
        # ... (manejo de errores igual que antes)
        raise HTTPException(status_code=500, detail=str(e))