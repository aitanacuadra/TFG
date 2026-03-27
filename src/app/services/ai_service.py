import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from google import genai
import json
import logging


from app.core.config import settings

FAISS_PATH = settings.FAISS_INDEX_PATH

print("Cargando base de datos vectorial y modelos en RAM (solo al arrancar)...")

embeddings_model = OllamaEmbeddings(model=settings.EMBEDDINGS_MODEL)

vectorstore = FAISS.load_local(FAISS_PATH, embeddings_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0)



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
    
    print("3. Generando metadatos finales con Gemma...")
    return chain.invoke({
        "context": contexto_normativo,
        "profile": profile, 
        "sample": sample_csv
    })


def get_simple_chat(prompt: str, model: str = settings.OLLAMA_MODEL) -> str:
    """Envía un prompt simple a Ollama y devuelve la respuesta en texto."""
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]
