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
import re
from pathlib import Path
from typing import Dict, Any


# LangChain (Ollama)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama


load_dotenv() # carga las cosas en el fichero .env asi asi luego puedes leerlas con os.getenv
# app = FastAPI() #instancia FastAPI
BASE_DATASET_URL = os.getenv("BASE_DATASET_URL", "https://example.org/dataset")
BASE_DOWNLOAD_URL = os.getenv("BASE_DOWNLOAD_URL", "https://example.org/datasets")

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




def slugify(text: str) -> str:
    """
    Convierte un texto en un identificador URL-safe.
    """
    text = (text or "").strip().lower()
    # Sustituir cualquier cosa que no sea alfanumérico o guion por '-'
    text = re.sub(r"[^\w\-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-") or "dataset"


def infer_xsd_type(col_name: str, series: pd.Series, llm_type: Optional[str] = None) -> str:
    """
    Inferir un tipo XSD razonable combinando:
    - dtype de pandas
    - y, si existe, un tipo 'semántico' de LLM (id, fecha, booleano...).
    """
    # Si el LLM ya nos dio un tipo XSD válido, respétalo
    if isinstance(llm_type, str) and llm_type.startswith("xsd:"):
        return llm_type

    dtype_str = str(series.dtype).lower()
    name = col_name.lower()

    # Heurísticas por nombre
    if "date" in name or name.endswith("_fecha") or name.startswith("fecha_"):
        # Si el dtype también parece datetime, date
        if "datetime" in dtype_str:
            return "xsd:dateTime"
        return "xsd:date"

    if name.endswith("_id") or name == "id":
        return "xsd:integer" if dtype_str.startswith("int") else "xsd:string"

    if name.startswith("is_") or name.startswith("has_"):
        return "xsd:boolean"

    # Heurísticas por dtype
    if dtype_str.startswith("int"):
        return "xsd:integer"
    if dtype_str.startswith("float"):
        return "xsd:decimal"
    if "bool" in dtype_str:
        return "xsd:boolean"
    if "datetime" in dtype_str:
        return "xsd:dateTime"

    # Por defecto, string
    return "xsd:string"


def build_keywords(df: pd.DataFrame, title: str, description: str) -> list:
    """
    Construye una lista sencilla de keywords a partir del título y las columnas.
    """
    base = ["dataset"]
    # Algunas palabras del título
    base.extend([w.lower() for w in re.findall(r"\w+", title) if len(w) > 3][:5])
    # Nombres de columnas
    base.extend([str(c) for c in df.columns[:10]])

    # Eliminar duplicados preservando orden
    seen = set()
    keywords = []
    for k in base:
        if k not in seen:
            seen.add(k)
            keywords.append(k)
    return keywords


def build_dcat3_metadata(
    raw_meta: Dict[str, Any],
    df: pd.DataFrame,
    filename: str,
    content_type: str,
    file_size_bytes: int,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Envuelve los metadatos 'planos' (title, description, variables, notes)
    en un objeto DCAT-3 JSON-LD.
    """

    # -------------------------
    # 1) Título, descripción, notas
    # -------------------------
    title = raw_meta.get("title") or filename
    description = raw_meta.get("description") or ""
    notes = raw_meta.get("notes") or ""

    # -------------------------
    # 2) ID estable del dataset
    # -------------------------
    if not dataset_id:
        stem = Path(filename).stem  # "students.csv" -> "students"
        dataset_id = slugify(stem)

    # -------------------------
    # 3) Normalizar variables del LLM
    #    Esperamos algo tipo:
    #    "variables": [ {"name": "...", "type": "...", "description": "...", "example": ...}, ... ]
    # -------------------------
    variables_raw = raw_meta.get("variables") or []
    var_info: Dict[str, Dict[str, Any]] = {}

    if isinstance(variables_raw, list):
        for v in variables_raw:
            if not isinstance(v, dict):
                continue
            name = (
                v.get("name")
                or v.get("schema:name")
                or v.get("column")
                or v.get("col")
            )
            if not name:
                continue
            llm_type = (
                v.get("semantic_type")
                or v.get("type")
                or v.get("schema:valueType")
                or v.get("dtype")
            )
            desc = v.get("description") or v.get("schema:description")
            example = v.get("example")
            var_info[str(name)] = {
                "llm_type": llm_type,
                "description": desc,
                "example": example,
            }
    elif isinstance(variables_raw, dict):
        # Caso {nombre_columna: tipo}
        for name, vtype in variables_raw.items():
            var_info[str(name)] = {"llm_type": vtype, "description": None, "example": None}

    # -------------------------
    # 4) mediaType y formato a partir del content_type
    # -------------------------
    media_type = content_type or "application/octet-stream"

    if "csv" in media_type.lower():
        dist_format = "text/csv"
        dist_suffix = "csv"
    elif "json" in media_type.lower():
        dist_format = "application/json"
        dist_suffix = "json"
    else:
        # fallback genérico
        dist_format = media_type
        dist_suffix = "data"

    # -------------------------
    # 5) Construir schema:variableMeasured usando df + LLM
    # -------------------------
    variable_measured = []
    for col in df.columns:
        name = str(col)
        info = var_info.get(name, {})
        series = df[name]

        llm_type = info.get("llm_type")
        desc = info.get("description")
        xsd_type = infer_xsd_type(name, series, llm_type)

        if not desc:
            desc = f"Columna '{name}' del dataset."

        variable_measured.append(
            {
                "@type": "schema:PropertyValue",
                "schema:name": name,
                "schema:valueType": xsd_type,
                "schema:description": desc,
            }
        )

    # -------------------------
    # 6) Keywords a partir del título y columnas
    # -------------------------
    keywords = build_keywords(df, title, description)

    # -------------------------
    # 7) Construir objeto DCAT-3
    # -------------------------
    metadata_dcat = {
        "@context": {
            "dcat": "http://www.w3.org/ns/dcat#",
            "dct": "http://purl.org/dc/terms/",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "schema": "http://schema.org/",
            "ex": "https://example.org/terms#",
        },
        "@type": "dcat:Dataset",
        "@id": f"{BASE_DATASET_URL}/{dataset_id}",
        "dct:title": title,
        "dct:description": description,
        "dct:identifier": filename,
        "dct:language": "es",
        "dct:type": "ex:TabularDataset",
        "dct:format": dist_format,
        "dcat:keyword": keywords,
        "dcat:distribution": [
            {
                "@type": "dcat:Distribution",
                "@id": f"{BASE_DATASET_URL}/{dataset_id}/distribution/{dist_suffix}",
                "dct:title": f"Distribución de {title}",
                "dct:description": notes or f"Archivo de datos: {filename}",
                "dct:format": dist_format,
                "dcat:mediaType": media_type,
                "dcat:byteSize": file_size_bytes,
                "dcat:downloadURL": f"{BASE_DOWNLOAD_URL}/{filename}",
            }
        ],
        "schema:variableMeasured": variable_measured,
        "dct:provenance": "Metadatos generados automáticamente a partir del contenido del archivo por un servicio de extracción asistido por un modelo de lenguaje.",
    }

    return metadata_dcat


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
    "Eres un asistente especializado en documentación de datos tabulares. "
    "A partir de un perfil de un DataFrame de pandas y una pequeña muestra en CSV, "
    "debes devolver UN ÚNICO objeto JSON con la siguiente estructura lógica:\n\n"
    '- Clave "title": string\n'
    '- Clave "description": string\n'
    '- Clave "variables": lista de objetos con:\n'
    '    - "name": nombre EXACTO de la columna\n'
    '    - "semantic_type": tipo semántico (id, fecha, fecha_hora, cantidad, categoria, booleano, etc.)\n'
    '    - "description": descripción en español\n'
    '    - "example": un valor de ejemplo\n'
    '- Clave "notes": string\n\n'
    "No generes DCAT ni otros envoltorios, SOLO este objeto JSON. "
    "No uses tipos de pandas como 'int64', 'float64', 'object' o 'bool' en semantic_type. "
    "Responde SIEMPRE en español y devuelve exclusivamente JSON válido."
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
        run = Run(
            endpoint="/process",
            filename=file.filename or "",
            content_type=file.content_type or "",
            size_bytes=len(content),
            status="started",
        )
        s.add(run)
        s.commit()
        s.refresh(run)

    try:
        df, kind = sniff_dataframe(content, file.content_type or "")
        profile = dataframe_profile(df)
        sample = head_as_csv(df)

        # 1) metadatos "planos" generados por LLM
        metadata_raw = generate_metadata_with_langchain(profile, sample)

        # 2) envolver en DCAT-3
        metadata_dcat = build_dcat3_metadata(
            raw_meta=metadata_raw,
            df=df,
            filename=file.filename or "dataset",
            content_type=file.content_type or "application/octet-stream",
            file_size_bytes=len(content),
        )

        # guardar metadata + completar run
        with db_session() as s:
            s.add(
                RunMetadata(
                    run_id=run.id,
                    metadata_json=json.dumps(metadata_dcat, ensure_ascii=False),
                )
            )
            run.status = "completed"
            s.add(run)
            s.commit()

        out_name = (
            file.filename
            if kind == "json"
            else (
                file.filename.removesuffix(".csv") + "_metadata.json"
                if file.filename.endswith(".csv")
                else file.filename + "_metadata.json"
            )
        )

        return {
            "message": "Procesado con éxito",
            "input_kind": kind,
            "output_filename": out_name,
            "metadata": metadata_dcat,
        }

    except Exception as e:
        with db_session() as s:
            run.status = "error"
            run.error = str(e)
            s.add(run)
            s.commit()
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {e}")
