import os
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.config import settings

# Configuramos el logger profesional
logger = logging.getLogger(__name__)

FAISS_PATH = settings.FAISS_INDEX_PATH

# 1. SOLUCIÓN RED DOCKER: Leemos la URL de Ollama desde las variables de entorno
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

print("Inicializando modelos LLM y Embeddings...")

# 2. INYECTAMOS LA URL A LOS COMPONENTES DE LANGCHAIN
embeddings_model = OllamaEmbeddings(model=settings.EMBEDDINGS_MODEL, base_url=OLLAMA_URL)
llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0, base_url=OLLAMA_URL,format="json")

# 3. SOLUCIÓN ROBUSTEZ: Inicialización defensiva de FAISS
vectorstore = None
retriever = None

if os.path.exists(FAISS_PATH) and os.path.exists(os.path.join(FAISS_PATH, "index.faiss")):
    print("Cargando base de datos vectorial FAISS en RAM...")
    vectorstore = FAISS.load_local(FAISS_PATH, embeddings_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
else:
    logger.warning(f"Índice FAISS no encontrado en {FAISS_PATH}. El RAG no funcionará correctamente.")

SYSTEM_PROMPT = (
    "Eres un ingeniero de datos experto en la normativa europea DCAT-AP. "
    "Tu tarea es generar UN ÚNICO objeto JSON con metadatos a partir del perfil de un DataFrame y una muestra CSV. "
    "DEBES basarte estrictamente en la siguiente documentación normativa de DCAT-AP para elegir los nombres de las claves y vocabularios:\n\n"
    "--- DOCUMENTACIÓN RECUPERADA (CONTEXTO) ---\n"
    "{context}\n"
    "-------------------------------------------\n\n"
    "El JSON debe incluir: title, description, variables (name, semantic_type, description, example) y notes. "
    "Responde SIEMPRE en español y devuelve exclusivamente JSON válido."
) 

def generate_metadata_with_langchain(profile: dict, sample_csv: str) -> dict:
    # 4. SOLUCIÓN AL ESTADO DEL SISTEMA
    if not retriever:
        raise RuntimeError("El sistema RAG no está operativo por falta del índice FAISS.")
        
    print("1. Consultando la normativa en FAISS (RAM)...")
    query_busqueda = "Propiedades obligatorias y recomendadas de DCAT-AP para describir conjuntos de datos (datasets) y variables."
    docs_recuperados = retriever.invoke(query_busqueda)
    contexto_normativo = "\n\n".join([doc.page_content for doc in docs_recuperados])
    
    print("2. Construyendo el prompt para Ollama...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "TASK: Genera metadatos estructurados.\nPROFILE:\n{profile}\nSAMPLE:\n{sample}")
    ])
    
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    
    print(f"3. Generando metadatos finales con {settings.OLLAMA_MODEL}...")
    
   
    try:
        resultado = chain.invoke({
            "context": contexto_normativo,
            "profile": profile, 
            "sample": sample_csv
        })
        return resultado
    except Exception as e:
        logger.error(f"Fallo crítico al parsear el JSON de Ollama: {str(e)}")
        raise ValueError("El modelo de lenguaje no devolvió un JSON válido. Inténtelo de nuevo.")