from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, Tuple
from datetime import datetime
from sqlmodel import SQLModel, Field, create_engine, Session
from contextlib import contextmanager


import os, io, json
import pandas as pd
import ollama

# LangChain (Ollama)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama


load_dotenv() # carga las cosas en el fichero .env asi asi luego puedes leerlas con os.getenv
# app = FastAPI() #instancia FastAPI

  #clave API desde .env
app=FastAPI() 
 #diccionario con la clave API y el numero de creditos
API_KEY = os.getenv("API_KEY", "dev-key")  # valor por defecto si no hay .env
API_KEYS_CREDITS = {API_KEY: 100}
#esta función valida que una clave API esté activa y tenga créditos suficientes antes de permitir su uso
def verify_api_key(x_api_key: str = Header(...)) -> str:
    credits = API_KEYS_CREDITS.get(x_api_key, 0)
    if credits <= 0:
        raise HTTPException(status_code=403, detail="Invalid or expired API key")
    return x_api_key

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./runs.db") #Guarda el resultado en la variable Python DATABASE_URL.
#Luego, esa variable se usa para conectar a la base de datos.
#Conéctate a la base de datos que viene descrita por esta URL
engine = create_engine(DATABASE_URL, echo=False)#motor de base de datos lo que hace que SQLModel pueda interactuar con la base de datos 

# esta clase es una tabla en la base de datos para guardar las ejecuciones
class Run(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    endpoint: str                      # <-- añade
    filename: str
    content_type: str
    size_bytes: int
    provider: str = "ollama"
    model: str = "gemma3:1b"
    status: str                        # started | completed | error
    error: Optional[str] = None 

# esta clase es otra tabla para guardar los metadatos generados
class RunMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int
    metadata_json: str  # JSON como string


def init_db():
    SQLModel.metadata.create_all(engine) #crea las tablas que tienen la definicion de table=True
#metadata en SQLAlchemy (y por extensión en SQLModel) es un objeto que contiene toda la información sobre el esquema de la base de datos: las tablas, columnas, tipos de datos, relaciones y restricciones definidas a través de los modelos

@app.on_event("startup") #"startup" es un evento en FastAPI que se dispara cuando la aplicación comienza a arrancar.
def on_startup():
    init_db()
#se ejecutará automáticamente cuando la aplicación web esté iniciándose, antes de que comience a recibir solicitudes
#la función on_startup() llama a init_db() para crear las tablas en la base de datos si no existen aún. Esto asegura que la base de datos está lista y configurada para que la aplicación pueda trabajar con ella desde el inicio, sin intervención manual.    


@contextmanager
def db_session():
    with Session(engine) as session:
        yield session


#ayuda a entender que es lo que quiere la API
class PromptIn(BaseModel):
    prompt: str



#POST enviar una solicitud a la API, quieres enviar datos al servidor 
@app.post("/generate")
def generate_text(body: PromptIn, x_api_key: str = Depends(verify_api_key)):
    # consume 1 crédito
    API_KEYS_CREDITS[x_api_key] -= 1
    resp = ollama.chat(model="gemma3:1b",
                       messages=[{"role": "user", "content": body.prompt}])
    return {"response": resp["message"]["content"]} #devuelve la respuesta del modelo en formato JSON




#ARCHIVOS

#Esta función recibe un archivo en bytes y su tipo de contenido, e intenta averiguar si es un JSON o un CSV. Lo convierte a un DataFrame de pandas y devuelve el DataFrame junto con el tipo detectado, "json" o "csv"
def sniff_dataframe(file_bytes: bytes, content_type: str) -> Tuple[pd.DataFrame, str]:
    try:
        if content_type and "json" in content_type.lower():
            obj = json.loads(file_bytes.decode("utf-8"))
            if isinstance(obj, list):
                df = pd.DataFrame(obj)
            elif isinstance(obj, dict):
                df = pd.json_normalize(obj)
            else:
                raise ValueError("JSON no reconocido")
            return df, "json"
        # CSV por defecto
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df, "csv"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al cargar archivo: {e}")
    
#Es como un mini report de pandas para inspeccionar rápido los datos.
def dataframe_profile(df: pd.DataFrame) -> dict:
    profile = {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(dt) for c, dt in df.dtypes.items()},
        "null_counts": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "examples": {}
    }
    for c in df.columns:
        ex = next((x for x in df[c].tolist() if pd.notna(x)), None)
        profile["examples"][str(c)] = ex
    return profile


def head_as_csv(df: pd.DataFrame, n: int = 5) -> str:
    buf = io.StringIO()
    df.head(n).to_csv(buf, index=False)
    return buf.getvalue()

def inject_metadata_into_json(original_bytes: bytes, metadata: dict) -> bytes:
    obj = json.loads(original_bytes.decode("utf-8"))
    if isinstance(obj, dict):
        obj["metadata"] = metadata
        return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    return json.dumps({"data": obj, "metadata": metadata}, ensure_ascii=False, indent=2).encode("utf-8")



# 1. sniff_dataframe Lo convierte entero en un DataFrame
# 2. dataframe_profile Usa el DataFrame de antes y calcula cosas del dataset entero(filas, columnas, tipos, nulos…).
# 3.  head_as_csv(df) Convierte esa mini-tabla en texto CSV
# 4. El modelo genera metadatos basados en lo de antes


# ===========================================
# LangChain: generar metadatos con ChatOllama
# ===========================================
SYSTEM = (
    "You are a data documentation assistant. Given a dataset profile and a small sample, "
    "return a JSON with: title, description, variables (name, type, description, example), "
    "and notes. Respond in Spanish."
) 

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "TASK: Genera metadatos para el dataset.\n\n"
     "PROFILE:\n{profile}\n\n"
     "SAMPLE (CSV first rows):\n{sample}\n\n"
     "Devuelve SOLO un objeto JSON con esas claves.")
])
parser = JsonOutputParser()


def generate_metadata_with_langchain(profile: dict, sample_csv: str) -> dict:
    llm = ChatOllama(model="gemma3:1b", temperature=0)
    chain = prompt | llm | parser
    return chain.invoke({"profile": profile, "sample": sample_csv})

# =======================================
# Endpoint: subir archivo y metadatos
# =======================================
@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    x_api_key: str = Depends(verify_api_key),
):
    API_KEYS_CREDITS[x_api_key] -= 1
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    # crear run (started)
    with db_session() as s:
        run = Run(endpoint="/process", filename=file.filename or "",
                  content_type=file.content_type or "", size_bytes=len(content),
                  status="started")
        s.add(run); s.commit(); s.refresh(run)

    try:
        df, kind = sniff_dataframe(content, file.content_type or "")
        profile = dataframe_profile(df)
        sample = head_as_csv(df)
        metadata = generate_metadata_with_langchain(profile, sample)

        # guardar metadata + completar run
        with db_session() as s:
            s.add(RunMetadata(run_id=run.id, metadata_json=json.dumps(metadata, ensure_ascii=False)))
            run.status = "completed"
            s.add(run); s.commit()

        out_name = (
            file.filename if kind == "json"
            else (file.filename.removesuffix(".csv") + "_metadata.json"
                  if file.filename.endswith(".csv") else file.filename + "_metadata.json")
        )
        return {"message": "Procesado con éxito", "input_kind": kind,
                "output_filename": out_name, "metadata": metadata}

    except Exception as e:
        with db_session() as s:
            run.status = "error"; run.error = str(e); s.add(run); s.commit()
        raise


