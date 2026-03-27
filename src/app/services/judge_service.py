import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.core.config import settings


judge_llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL, 
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0
)


SYSTEM_PROMPT_JUDGE = """
Eres un Auditor Senior de data.europa.eu experto en la metodología MQA (Metadata Quality Assessment). 
Tu tarea es evaluar la calidad de los metadatos DCAT-AP generados a partir de un archivo tabular.

Debes aplicar rigurosamente las siguientes dimensiones de calidad descritas en el marco teórico:

1. FINDABILITY (Localización - Máx 100 pts): 
   - Evalúa si dcat:keyword y dcat:theme permiten encontrar el dataset.
   - Penaliza si las keywords son solo nombres de columnas.

2. INTEROPERABILIDAD (Máx 110 pts): 
   - Verifica el uso de vocabularios controlados en dct:format y dcat:mediaType[cite: 226].
   - Comprueba la conformidad con el perfil DCAT-AP 3.0.0[cite: 226].

3. REUSABILIDAD (Máx 75 pts): 
   - Presencia de información de licencias (dct:license) y procedencia (dct:provenance)[cite: 227, 280].

4. EXACTITUD Y VALIDEZ (Dimensiones DAMA):
   - ¿Reflejan los metadatos la realidad de la muestra? (Exactitud) [cite: 252].
   - ¿Cumplen los tipos xsd:dateTime o xsd:integer con los datos reales? (Validez) [cite: 256].

Debes devolver EXCLUSIVAMENTE un JSON con esta estructura:
{{
  "score_global": 0, (Escala 0-100 basada en pesos MQA)
  "categoria_mqa": "Excelente | Buena | Suficiente | Mala", (Según rangos MQA: 351-405, 221-350, 121-220, 0-120) 
  "analisis_fair": {{
    "findability": "puntuación y crítica",
    "interoperability": "puntuación y crítica",
    "reusability": "puntuación y crítica",
    "exactitud_validez": "puntuación y crítica"
  }},
  "errores_gobernanza": ["lista de faltas de consistencia o exactitud detectadas"],
  "mejoras_prioritarias": ["puntos clave para alcanzar la categoría Excelente"]
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

parser = JsonOutputParser()

def evaluate_metadata_with_gemini(profile: dict, sample_csv: str, metadata: dict) -> dict:
    if not settings.GEMINI_API_KEY:
        return {
            "score_global": 0,
            "categoria_mqa": "Error",
            "analisis_fair": "Falta configuración de API Key."
        }

    chain = JUDGE_PROMPT | judge_llm | parser
    return chain.invoke({
        "profile": profile,
        "sample": sample_csv,
        "metadata": metadata
    })