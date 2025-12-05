import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from app.core.config import settings

# =============================================================================
# 1. FUNCIONES UTILITARIAS (Herramientas genéricas)
# =============================================================================

def slugify(text: str) -> str:
    """Convierte texto a formato URL-safe (ej: 'Hola Mundo' -> 'hola-mundo')."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-") or "dataset"

def infer_xsd_type(col_name: str, series: pd.Series, llm_type: Optional[str] = None) -> str:
    """Infiere el tipo de dato XSD (string, integer, date) para una columna."""
    # Si la IA ya nos dio un tipo XSD, lo usamos
    if isinstance(llm_type, str) and llm_type.startswith("xsd:"):
        return llm_type

    dtype_str = str(series.dtype).lower()
    name = col_name.lower()

    # Heurísticas por nombre
    if "date" in name or name.startswith("fecha"):
        return "xsd:dateTime" if "datetime" in dtype_str else "xsd:date"
    if name.endswith("_id") or name == "id":
        return "xsd:integer" if dtype_str.startswith("int") else "xsd:string"
    if name.startswith("is_") or name.startswith("has_"):
        return "xsd:boolean"

    # Heurísticas por tipo de dato interno
    if dtype_str.startswith("int"): return "xsd:integer"
    if dtype_str.startswith("float"): return "xsd:decimal"
    if "bool" in dtype_str: return "xsd:boolean"
    if "datetime" in dtype_str: return "xsd:dateTime"

    return "xsd:string"

def build_keywords(df: pd.DataFrame, title: str) -> List[str]:
    """Genera keywords a partir del título y las columnas."""
    base = ["dataset"]
    base.extend([w.lower() for w in re.findall(r"\w+", title) if len(w) > 3][:5])
    base.extend([str(c) for c in df.columns[:10]])
    
    # Eliminar duplicados preservando orden
    return list(dict.fromkeys(base))

# =============================================================================
# 2. FUNCIONES AUXILIARES PRIVADAS (Detalles de implementación)
# =============================================================================

def _normalize_llm_variables(variables_raw: Any) -> Dict[str, Dict[str, Any]]:
    """Limpia y estandariza la salida de variables que viene de la IA."""
    var_info = {}
    
    if isinstance(variables_raw, list):
        for v in variables_raw:
            if not isinstance(v, dict): continue
            # Buscar nombre en varias claves posibles
            name = v.get("name") or v.get("column") or v.get("col")
            if not name: continue
            
            var_info[str(name)] = {
                "llm_type": v.get("semantic_type") or v.get("type"),
                "description": v.get("description"),
                "example": v.get("example"),
            }
            
    elif isinstance(variables_raw, dict):
        for name, vtype in variables_raw.items():
            var_info[str(name)] = {"llm_type": vtype, "description": None}
            
    return var_info

def _determine_distribution_format(content_type: str) -> Tuple[str, str]:
    """Decide el formato y extensión basado en el content-type."""
    ct = (content_type or "").lower()
    if "csv" in ct:
        return "text/csv", "csv"
    if "json" in ct:
        return "application/json", "json"
    return ct or "application/octet-stream", "data"

def _build_measured_variables(df: pd.DataFrame, var_info: Dict) -> List[Dict]:
    """Crea la lista de schema:variableMeasured recorriendo el DataFrame."""
    variables = []
    for col in df.columns:
        name = str(col)
        info = var_info.get(name, {})
        series = df[name]

        xsd_type = infer_xsd_type(name, series, info.get("llm_type"))
        desc = info.get("description") or f"Columna '{name}' del dataset."

        variables.append({
            "@type": "schema:PropertyValue",
            "schema:name": name,
            "schema:valueType": xsd_type,
            "schema:description": desc,
        })
    return variables

# =============================================================================
# 3. FUNCIÓN CEREBRO (Orquestador Principal)
# =============================================================================

def build_dcat3_metadata(
    raw_meta: Dict[str, Any],
    df: pd.DataFrame,
    filename: str,
    content_type: str,
    file_size_bytes: int,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Función Principal (Cerebro): Orquesta la creación del JSON-LD.
    No contiene lógica compleja, solo llama a los especialistas.
    """
    
    # 1. Extracción de datos básicos
    title = raw_meta.get("title") or filename
    description = raw_meta.get("description") or ""
    notes = raw_meta.get("notes") or ""
    
    # 2. Generación de ID y Keywords
    if not dataset_id:
        dataset_id = slugify(Path(filename).stem)
    keywords = build_keywords(df, title)

    # 3. Procesamiento de detalles técnicos (Llamadas a funciones privadas)
    var_info = _normalize_llm_variables(raw_meta.get("variables"))
    dist_format, dist_suffix = _determine_distribution_format(content_type)
    variable_measured = _build_measured_variables(df, var_info)

    # 4. Ensamblaje final del JSON (Structure building)
    return {
        "@context": {
            "dcat": "http://www.w3.org/ns/dcat#",
            "dct": "http://purl.org/dc/terms/",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "schema": "http://schema.org/",
            "ex": "https://example.org/terms#",
        },
        "@type": "dcat:Dataset",
        "@id": f"{settings.BASE_DATASET_URL}/{dataset_id}",
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
                "@id": f"{settings.BASE_DATASET_URL}/{dataset_id}/distribution/{dist_suffix}",
                "dct:title": f"Distribución de {title}",
                "dct:description": notes or f"Archivo de datos: {filename}",
                "dct:format": dist_format,
                "dcat:mediaType": content_type or "application/octet-stream",
                "dcat:byteSize": file_size_bytes,
                "dcat:downloadURL": f"{settings.BASE_DOWNLOAD_URL}/{filename}",
            }
        ],
        "schema:variableMeasured": variable_measured,
        "dct:provenance": "Metadatos generados automáticamente mediante IA.",
    }