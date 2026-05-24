import logging
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore # Sustituimos FAISS por Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings

# Configuramos el logger profesional
logger = logging.getLogger(__name__)

logger.info("Inicializando modelos LLM y Embeddings...")

# 1. INYECTAMOS LA URL A LOS COMPONENTES DE LANGCHAIN (Usando estrictamente Pydantic Settings)
embeddings_model = OllamaEmbeddings(
    model=settings.EMBEDDINGS_MODEL, 
    base_url=settings.OLLAMA_BASE_URL
)

llm = ChatOllama(
    model=settings.OLLAMA_MODEL, 
    temperature=0, 
    base_url=settings.OLLAMA_BASE_URL,
    format="json" # Forzamos la salida estructurada
)

# 2. INICIALIZACIÓN ROBUSTA DE QDRANT
retriever = None
try:
    logger.info(f"Conectando al motor vectorial Qdrant en {settings.QDRANT_URL}...")
    client = QdrantClient(url=settings.QDRANT_URL)
    
    # Verificamos si la colección existe; si no, el retriever quedará desactivado hasta que se ingeste la normativa
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

# 3. PROMPT ENGINEERING ESTRICTO
SYSTEM_PROMPT = (
    "Eres un ingeniero de datos experto en la normativa europea DCAT-AP. "
    "Tu tarea es generar UN ÚNICO objeto JSON con metadatos a partir del perfil estructural de un dataset y una muestra de datos. "
    "DEBES basarte estrictamente en la siguiente documentación normativa recuperada:\n\n"
    "--- CONTEXTO NORMATIVO DCAT-AP ---\n"
    "{context}\n"
    "----------------------------------\n\n"
    "El JSON debe incluir: 'title', 'description', 'keyword' (lista de strings) y 'theme'. "
    "Las palabras clave DEBEN ser específicas del dominio del dataset. NUNCA uses términos genéricos como 'dataset', 'data' o 'metadata'. "
    "El 'theme' debe corresponder al vocabulario oficial de temas de datos de la UE.\n\n"
    "IMPORTANTE: El siguiente ejemplo muestra ÚNICAMENTE el formato esperado. "
    "El contenido que generes debe inferirse SIEMPRE del perfil y la muestra del dataset real que recibirás, NUNCA del ejemplo.\n\n"
    "EJEMPLO DE FORMATO (no copies el contenido):\n"
    "Dataset: columnas ['municipio', 'fecha', 'lluvia_mm', 'temperatura_max', 'temperatura_min']\n"
    "JSON resultante:\n"
    "{{\n"
    "  \"title\": \"Datos meteorológicos diarios por municipio\",\n"
    "  \"description\": \"Registros diarios de temperatura y precipitaciones por municipio español.\",\n"
    "  \"keyword\": [\"meteorología\", \"temperatura\", \"precipitación\", \"municipio\", \"clima\"],\n"
    "  \"theme\": \"http://publications.europa.eu/resource/authority/data-theme/ENVI\"\n"
    "}}\n\n"
    "Responde SIEMPRE en español y devuelve exclusivamente JSON válido."
)

def generate_embeddings(text: str) -> list[float]:
    """
    Genera embeddings usando el modelo local configurado.
    Vital para insertar en Qdrant de forma manual.
    """
    try:
        # Usamos el modelo de Langchain directamente para embeddir la query
        return embeddings_model.embed_query(text)
    except Exception as e:
        logger.error(f"Error al generar embeddings: {e}")
        raise ValueError("No se pudo generar el vector de embeddings.")

def generate_metadata_with_langchain(profile: dict, sample_csv: str) -> dict:
    if not retriever:
        raise RuntimeError("El sistema RAG no está operativo por falta de conexión a Qdrant o colección vacía.")
        
    logger.info("Consultando la normativa DCAT-AP en Qdrant...")
    query_busqueda = "Propiedades obligatorias y recomendadas de DCAT-AP para describir conjuntos de datos (datasets)."
    
    # RAG: Recuperación de documentos
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