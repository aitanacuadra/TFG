import logging
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore # Sustituimos FAISS por Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings


logger = logging.getLogger(__name__)
logger.info("Inicializando modelos LLM y Embeddings...")

embeddings_model = OllamaEmbeddings(
    model=settings.EMBEDDINGS_MODEL, 
    base_url=settings.OLLAMA_BASE_URL
)

llm = ChatOllama(
    model=settings.OLLAMA_MODEL, 
    temperature=0, 
    base_url=settings.OLLAMA_BASE_URL,
    format="json" 
)

retriever = None
try:
    logger.info(f"Conectando al motor vectorial Qdrant en {settings.QDRANT_URL}...")
    client = QdrantClient(url=settings.QDRANT_URL)
    
    if client.collection_exists(settings.QDRANT_COLLECTION):
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=embeddings_model,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        logger.info("Retriever de Qdrant configurado correctamente.")
    else:
        logger.warning(f"La colección '{settings.QDRANT_COLLECTION}' no existe en Qdrant. Debes ingestar la normativa DCAT-AP primero.")
except Exception as e:
    logger.error(f"Fallo crítico al conectar con Qdrant: {e}")

SYSTEM_PROMPT = (
    "Eres un ingeniero de datos experto en la normativa europea DCAT-AP. "
    "Tu tarea es generar UN ÚNICO objeto JSON con metadatos a partir del perfil estructural de un dataset y una muestra de datos. "
    "DEBES basarte estrictamente en la siguiente documentación normativa recuperada:\n\n"
    "--- CONTEXTO NORMATIVO DCAT-AP ---\n"
    "{context}\n"
    "----------------------------------\n\n"
    "El JSON debe incluir: 'title', 'description', 'keyword' (lista de strings) y 'theme'. "
    "Responde SIEMPRE en español y devuelve exclusivamente JSON válido."
)

def generate_embeddings(text: str) -> list[float]:
    """
    Genera embeddings usando el modelo local configurado.
    Vital para insertar en Qdrant de forma manual.
    """
    try:
        return embeddings_model.embed_query(text)
    except Exception as e:
        logger.error(f"Error al generar embeddings: {e}")
        raise ValueError("No se pudo generar el vector de embeddings.")

def generate_metadata_with_langchain(profile: dict, sample_csv: str) -> dict:
    if not retriever:
        raise RuntimeError("El sistema RAG no está operativo por falta de conexión a Qdrant o colección vacía.")
        
    logger.info("Consultando la normativa DCAT-AP en Qdrant...")
    query_busqueda = "Propiedades obligatorias y recomendadas de DCAT-AP para describir conjuntos de datos (datasets)."
    docs_recuperados = retriever.invoke(query_busqueda)
    contexto_normativo = "\n\n".join([doc.page_content for doc in docs_recuperados])
    
    logger.info("Construyendo la cadena de Prompting para Ollama...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Genera los metadatos estructurados para el siguiente dataset.\n\nPERFIL (Esquema):\n{profile}\n\nMUESTRA DE DATOS:\n{sample}")
    ])
    
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    
    logger.info(f"Generando metadatos finales con {settings.OLLAMA_MODEL}...")
    try:
        resultado = chain.invoke({
            "context": contexto_normativo,
            "profile": profile, 
            "sample": sample_csv
        })
        logger.info("Metadatos generados exitosamente.")
        return resultado
    except Exception as e:
        logger.error(f"Fallo crítico en la generación o parseo del JSON: {str(e)}", exc_info=True)
        raise ValueError("El modelo de lenguaje falló al generar la respuesta esperada.")