import logging
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.core.config import settings

# 1. Configuración profesional del Logger (¡Adiós prints!)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOC_PATH = settings.RAG_DOC_PATH

def init_vector_db():
    logger.info(f"Cargando documento normativo desde {DOC_PATH}...")
    try:
        # Usamos el loader para extraer el texto limpio del HTML
        loader = BSHTMLLoader(DOC_PATH, bs_kwargs={'features': 'html.parser'})
        document = loader.load()
    except Exception as e:
        logger.error(f"Error crítico al leer el documento: {e}")
        return
    
    logger.info("Dividiendo el documento en fragmentos (chunks)...")
    # Estrategia de fragmentación para mantener el contexto semántico sin ahogar al LLM
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(document)
    
    logger.info(f"Conectando a Qdrant en {settings.QDRANT_URL}...")
    
    # 2. Inicialización estricta del modelo de embeddings usando variables de entorno seguras
    embeddings_model = OllamaEmbeddings(
        model=settings.EMBEDDINGS_MODEL,
        base_url=settings.OLLAMA_BASE_URL
    )

    logger.info(f"Vectorizando e ingestando {len(chunks)} fragmentos en la colección '{settings.QDRANT_COLLECTION}'...")
    
    try:
        # 3. Creación e ingesta directa en el motor vectorial Qdrant
        # force_recreate=True permite ejecutar este script para actualizar la normativa limpiando lo anterior
        QdrantVectorStore.from_documents(
            chunks,
            embeddings_model,
            url=settings.QDRANT_URL,
            collection_name=settings.QDRANT_COLLECTION,
            force_recreate=True 
        )
        logger.info("¡Base de datos vectorial Qdrant inicializada y poblada con éxito!")
    except Exception as e:
        logger.error(f"Fallo al conectar o insertar en Qdrant: {e}")

if __name__ == "__main__":
    init_vector_db()