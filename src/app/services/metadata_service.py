import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from app.core.config import settings

def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-") or "dataset"

def infer_xsd_type(col_name: str, series: pd.Series, llm_type: Optional[str] = None) -> str:
    if isinstance(llm_type, str) and llm_type.startswith("xsd:"):
        return llm_type

    dtype_str = str(series.dtype).lower()
    name = col_name.lower()

    if "date" in name or name.startswith("fecha"):
        return "xsd:dateTime" if "datetime" in dtype_str else "xsd:date"
    if name.endswith("_id") or name == "id":
        return "xsd:integer" if dtype_str.startswith("int") else "xsd:string"
    if name.startswith("is_") or name.startswith("has_"):
        return "xsd:boolean"

    if dtype_str.startswith("int"): return "xsd:integer"
    if dtype_str.startswith("float"): return "xsd:decimal"
    if "bool" in dtype_str: return "xsd:boolean"
    if "datetime" in dtype_str: return "xsd:dateTime"

    return "xsd:string"

def _build_fallback_keywords(df: pd.DataFrame, title: str) -> List[str]:
    """Genera keywords programáticas SOLO si el LLM falla."""
    base = ["dataset", "tabular"]
    base.extend([w.lower() for w in re.findall(r"\w+", title) if len(w) > 3][:3])
    return list(dict.fromkeys(base))

def _normalize_llm_variables(variables_raw: Any) -> Dict[str, Dict[str, Any]]:
    var_info = {}
    if isinstance(variables_raw, list):
        for v in variables_raw:
            if not isinstance(v, dict): continue
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

def _determine_distribution_format(content_type: str, file_format: Optional[str] = None) -> Tuple[str, str]:
    fmt = (file_format or "").lower()
    if fmt == "csv":
        return "text/csv", "csv"
    if fmt == "json":
        return "application/json", "json"
    ct = (content_type or "").lower()
    if "csv" in ct:
        return "text/csv", "csv"
    if "json" in ct:
        return "application/json", "json"
    return ct or "application/octet-stream", "data"

def _build_measured_variables(df: pd.DataFrame, var_info: Dict) -> List[Dict]:
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

def build_dcat3_metadata(
    raw_meta: Dict[str, Any],
    df: pd.DataFrame,
    filename: str,
    content_type: str,
    file_size_bytes: int,
    dataset_id: Optional[str] = None,
    file_format: Optional[str] = None,
) -> Dict[str, Any]:
    
    title = raw_meta.get("title") or filename
    description = raw_meta.get("description") or "Dataset generado automáticamente."
    notes = raw_meta.get("notes") or ""
    llm_keywords = raw_meta.get("keyword") or raw_meta.get("keywords")

    if isinstance(llm_keywords, list) and len(llm_keywords) > 0:
        keywords = [str(k).lower() for k in llm_keywords]
    else:
        keywords = _build_fallback_keywords(df, title)
        
    theme = raw_meta.get("theme") or "http://publications.europa.eu/resource/authority/data-theme/TECH"
    
    
    if not dataset_id:
        dataset_id = slugify(Path(filename).stem)

    
    var_info = _normalize_llm_variables(raw_meta.get("variables"))
    dist_format, dist_suffix = _determine_distribution_format(content_type, file_format)
    variable_measured = _build_measured_variables(df, var_info)

    
    return {
        "@context": {
            "dcat": "http://www.w3.org/ns/dcat#",
            "dct": "http://purl.org/dc/terms/",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "schema": "http://schema.org/",
            "foaf": "http://xmlns.com/foaf/0.1/", 
        },
        "@type": "dcat:Dataset",
        "@id": f"{settings.BASE_DATASET_URL}/{dataset_id}",
        "dct:title": title,
        "dct:description": description,
        "dcat:theme": theme, 
        "dct:identifier": filename,
        "dct:language": "es",
        "dct:type": "http://purl.org/dc/dcmitype/Dataset", 
        "dct:format": dist_format,
        "dcat:keyword": keywords,
        "dcat:distribution": [
            {
                "@type": "dcat:Distribution",
                "@id": f"{settings.BASE_DATASET_URL}/{dataset_id}/distribution/{dist_suffix}",
                "dct:title": f"Distribución de {title}",
                "dct:description": notes or f"Archivo de datos descargable: {filename}",
                "dct:format": dist_format,
                "dcat:mediaType": content_type or "application/octet-stream",
                "dcat:byteSize": file_size_bytes,
                "dcat:downloadURL": f"{settings.BASE_DOWNLOAD_URL}/{filename}",
            }
        ],
        "schema:variableMeasured": variable_measured,
        "dct:provenance": "Metadatos semánticos inferidos mediante Inteligencia Artificial (LLM).",
    }