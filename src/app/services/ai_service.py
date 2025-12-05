from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

SYSTEM_PROMPT = (
    "Eres un asistente especializado en documentación de datos tabulares. "
    "A partir de un perfil de un DataFrame y una muestra CSV, devuelve UN ÚNICO objeto JSON "
    "con claves: title, description, variables (name, semantic_type, description, example) y notes. "
    "Responde SIEMPRE en español y devuelve exclusivamente JSON válido."
)

def generate_metadata_with_langchain(profile: dict, sample_csv: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "TASK: Genera metadatos.\nPROFILE:\n{profile}\nSAMPLE:\n{sample}")
    ])
    parser = JsonOutputParser()
    llm = ChatOllama(model="gemma3:1b", temperature=0)
    chain = prompt | llm | parser
    return chain.invoke({"profile": profile, "sample": sample_csv})

def get_simple_chat(prompt: str, model: str = "gemma3:1b") -> str:
    """Envía un prompt simple a Ollama y devuelve la respuesta en texto."""
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]