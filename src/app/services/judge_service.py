import re
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings

def _parse_llm_json(text: str) -> dict:
    text = re.sub(r'```json\s*|\s*```', '', text).strip()
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return json.loads(text)


judge_llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL, 
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0
)


SYSTEM_PROMPT_JUDGE = """
Eres un Auditor Senior de data.europa.eu experto en la metodología MQA (Metadata Quality Assessment). 
Tu tarea es evaluar la calidad de los metadatos DCAT-AP generados a partir de un archivo tabular.

Debes aplicar rigurosamente las siguientes 5 dimensiones de calidad:

1. FINDABILITY (Máx 100 pts): 
   - Evalúa el uso de palabras clave (dcat:keyword), temas (dcat:theme), localización (dct:spatial) y temporalidad (dct:temporal).
2. ACCESSIBILITY (Máx 100 pts): 
   - Evalúa la presencia y accesibilidad de dcat:accessURL y dcat:downloadURL. 
3. INTEROPERABILIDAD (Máx 130 pts aprox): 
   - Verifica formatos abiertos, estandarizados y legibles por máquina (dct:format / dcat:mediaType). 
   - Comprueba conformidad con DCAT-AP 3.0.0.
4. REUSABILIDAD (Máx 75 pts): 
   - Presencia de licencias (dct:license), restricciones (dct:accessRights), contacto (dcat:contactPoint) y editor (dct:publisher).
5. CONTEXTUALITY (Máx 20 pts):
   - Presencia de metadatos de contexto: derechos (dct:rights), tamaño (dcat:byteSize), y fechas (dct:issued, dct:modified).

Debes devolver EXCLUSIVAMENTE un JSON con esta estructura:
{{
  "score_global": 0, 
  "categoria_mqa": "Excelente (351-405) | Buena (221-350) | Suficiente (121-220) | Mala (0-120)",
  "analisis_detallado": {{
    "findability": "puntuación y crítica",
    "accessibility": "puntuación y crítica",
    "interoperability": "puntuación y crítica",
    "reusability": "puntuación y crítica",
    "contextuality": "puntuación y crítica"
  }},
  "mejoras_prioritarias": ["lista de acciones para subir de categoría"]
}}
"""


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_JUDGE),
    ("human", """
MUESTRA DEL DATASET (CSV):
{sample}

PERFIL TÉCNICO:
{profile}

METADATOS GENERADOS (JSON-LD):
{metadata}
""")
])

def evaluate_metadata_with_gemini(profile: dict, sample_csv: str, metadata: dict) -> dict:
    if not settings.GEMINI_API_KEY:
        return {
            "score_global": 0,
            "categoria_mqa": "Error",
            "analisis_fair": "Falta configuración de API Key."
        }

    chain = JUDGE_PROMPT | judge_llm | StrOutputParser()
    raw = chain.invoke({
        "profile": profile,
        "sample": sample_csv,
        "metadata": metadata
    })
    return _parse_llm_json(raw)