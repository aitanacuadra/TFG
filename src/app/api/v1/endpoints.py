import json
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Header
from sqlmodel import Session
from pydantic import BaseModel  

from app.db.session import get_db
from app.models.run import Run, RunMetadata
from app.services import file_service, ai_service, metadata_service
from app.core.config import settings
from app.api.deps import verify_api_key
from app.schemas import ProcessFileResponse

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
    
    response_text = ai_service.get_simple_chat(body.prompt)
    return {"response": response_text}


@router.post(
    "/process",

    response_model=ProcessFileResponse,  
    summary="Procesa un archivo y genera metadatos DCAT",
    description="Sube un CSV o JSON para obtener sus metadatos en formato JSON-LD compatible con DCAT-AP."
) 

async def process_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Archivo vacío")

   
    run = Run(endpoint="/process", filename=file.filename, content_type=file.content_type, size_bytes=len(content))
    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        # 1. Analizar archivo 
        df, kind = file_service.sniff_dataframe(content, file.content_type or "")
        profile = file_service.dataframe_profile(df)
        sample = file_service.head_as_csv(df)
        
        # 2. IA genera metadatos crudos
        metadata_raw = ai_service.generate_metadata_with_langchain(profile, sample)
        
       
        metadata_dcat = metadata_service.build_dcat3_metadata(
            raw_meta=metadata_raw,
            df=df,
            filename=file.filename or "dataset",
            content_type=file.content_type or "application/octet-stream",
            file_size_bytes=len(content)
        )
       

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
            "metadata": metadata_dcat  
        }

    except Exception as e:
  
        raise HTTPException(status_code=500, detail=str(e))