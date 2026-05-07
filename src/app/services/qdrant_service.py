import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings

logger = logging.getLogger(__name__)

# Inicializamos el cliente apuntando al contenedor de Docker
client = QdrantClient(url=settings.QDRANT_URL)

def upsert_dataset(vector: list[float], payload: dict, dataset_id: str):
    """
    Inserta o actualiza los metadatos de un dataset en Qdrant.
    """
    try:
        # Qdrant requiere que los IDs sean enteros o UUIDs. 
        # Generamos un UUID determinista a partir del ID de tu base de datos SQL
        point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, str(dataset_id)))
        
        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload # Aquí va tu JSON de DCAT-AP intacto
        )
        
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=[point]
        )
        logger.info(f"Metadatos del dataset {dataset_id} vectorizados e indexados en Qdrant correctamente.")
        
    except Exception as e:
        logger.error(f"Error crítico al insertar en Qdrant el dataset {dataset_id}: {e}")
        raise