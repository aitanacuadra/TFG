import os
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from app.core.config import settings


FAISS_PATH = settings.FAISS_INDEX_PATH
DOC_PATH = settings.RAG_DOC_PATH

def init_vector_db():
    print("1. Cargando la normativa DCAT-AP en HTML...")
    loader = BSHTMLLoader(DOC_PATH, bs_kwargs={'features': 'html.parser'})
    document = loader.load()
    
    print("2. Dividiendo el documento en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(document)
    print(f"Se han generado {len(chunks)} fragmentos.")
    
    print("3. Creando embeddings con Ollama y guardando en FAISS...")
   
    embeddings_model = OllamaEmbeddings(model=settings.EMBEDDINGS_MODEL)
    
    db = FAISS.from_documents(chunks, embeddings_model)
    db.save_local(FAISS_PATH)
    print(f"¡Base de datos vectorial creada con éxito en la carpeta '{FAISS_PATH}'!")

if __name__ == "__main__":
    init_vector_db()