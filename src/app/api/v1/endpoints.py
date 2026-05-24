import json
import logging # <- ¡Obligatorio para el 10!
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlmodel import Session
from pydantic import BaseModel  
from app.db.session import get_db
from app.models.run import Run, RunMetadata
from app.services import file_service, ai_service, metadata_service, judge_service
# Asumo que crearás un qdrant_service para manejar la inserción
from app.services import qdrant_service 
from app.core.config import settings
from app.api.deps import verify_api_key
from app.schemas import ProcessFileResponse

logger = logging.getLogger(__name__)
router = APIRouter()

class PromptIn(BaseModel):
    prompt: str

@router.post(
    "/process",
    response_model=ProcessFileResponse,  
    summary="Procesa un archivo, genera metadatos DCAT-AP y los evalua",
    description="Sube un CSV o JSON para obtener sus metadatos compatibles con DCAT-AP"
) 
async def process_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    content = await file.read()
    MAX_FILE_SIZE = 10 * 1024 * 1024  

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="El archivo excede el límite de 10MB.")

    if not content:
        raise HTTPException(status_code=400, detail="Archivo vacío")
   
    run = Run(endpoint="/process", filename=file.filename, content_type=file.content_type, size_bytes=len(content))
    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        df, file_format = file_service.sniff_dataframe(content, file.content_type or "")
        profile = file_service.dataframe_profile(df)
        sample = file_service.head_as_csv(df)
        
        
        metadata_raw = ai_service.generate_metadata_with_langchain(profile, sample)
        
        
        metadata_dcat = metadata_service.build_dcat3_metadata(
            raw_meta=metadata_raw,
            df=df,
            filename=file.filename or "dataset",
            content_type=file.content_type or "application/octet-stream",
            file_size_bytes=len(content),
            file_format=file_format
        )
        
        
        judge_result = judge_service.evaluate_metadata_with_gemini(
            profile=profile,
            sample_csv=sample,
            metadata=metadata_dcat
        )   
       
        text_to_embed = f"{metadata_dcat.get('title', '')} {metadata_dcat.get('description', '')} {' '.join(metadata_dcat.get('keyword', []))}"
        
        
        embedding_vector = ai_service.generate_embeddings(text_to_embed)
        
        
        qdrant_service.upsert_dataset(
            vector=embedding_vector,
            payload=metadata_dcat, 
            dataset_id=str(run.id) 
        )
        
        
        db.add(RunMetadata(
            run_id=run.id,
            metadata_json=json.dumps(metadata_dcat, ensure_ascii=False),
            evaluation_json=json.dumps(judge_result, ensure_ascii=False)
        ))
        run.status = "completed"
        db.add(run)
        db.commit()

        return {
            "message": "Procesado y vectorizado con éxito",
            "output_filename": f"{file.filename}_meta.json",
            "metadata": metadata_dcat,
            "evaluation": judge_result 
        }
    
    except Exception as e:
        
        logger.error(f"Error procesando el archivo {file.filename} en el Run {run.id}: {str(e)}", exc_info=True)
        
        run.status = "failed"
        db.add(run)
        db.commit()
        raise HTTPException(status_code=500, detail="Error interno al procesar el archivo. Revisa los logs.")