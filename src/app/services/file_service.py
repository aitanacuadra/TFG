import io
import json
import pandas as pd
from typing import Tuple, Dict
from fastapi import HTTPException


MAX_SAMPLE_ROWS = 100 

def sniff_dataframe(file_bytes: bytes, content_type: str) -> Tuple[pd.DataFrame, str]:
    
    try:
        inicio = file_bytes[:50].decode('utf-8', errors='ignore').strip()
        
        if inicio.startswith("{") or inicio.startswith("["):
            texto_completo = file_bytes.decode('utf-8')
            obj = json.loads(texto_completo)
            if isinstance(obj, list):
                obj_sample = obj[:MAX_SAMPLE_ROWS]
                df = pd.DataFrame(obj_sample)
            else:
                df = pd.json_normalize(obj).head(MAX_SAMPLE_ROWS)
            return df, "json"
            
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error en JSON. Detalle técnico: {str(e)}"
        )
    
    encodings = ["utf-8", "latin-1"]
    separators = [",", ";", "\t", "|"]
    sample_bytes = file_bytes[:10240] 
    mejor_encoding = None
    mejor_sep = None
    
    for enc in encodings:
        for sep in separators:
            try:
                df_test = pd.read_csv(io.BytesIO(sample_bytes), encoding=enc, sep=sep, nrows=5)
                if df_test.shape[1] > 1:  
                    mejor_encoding = enc
                    mejor_sep = sep
                    break
            except Exception:
                continue
        if mejor_encoding:
            break 

    try:
        if mejor_encoding and mejor_sep:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=mejor_encoding, sep=mejor_sep, on_bad_lines='skip', nrows=MAX_SAMPLE_ROWS)
        else:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1", sep=None, engine="python", nrows=MAX_SAMPLE_ROWS)
            
        return df, "csv"

    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"No se pudo procesar el CSV. Error técnico: {str(e)}"
        )
    
def dataframe_profile(df: pd.DataFrame) -> Dict:
    profile = {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(dt) for c, dt in df.dtypes.items()},
        "null_counts": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "examples": {}
    }
    for c in df.columns:
        
        try:
            valid_series = df[c].dropna()
            if not valid_series.empty:
                profile["examples"][str(c)] = valid_series.iloc[0]
            else:
                profile["examples"][str(c)] = None
        except Exception:
            profile["examples"][str(c)] = None
            
    return profile

def head_as_csv(df: pd.DataFrame, n: int = 5) -> str:
    buf = io.StringIO()
    df.head(n).to_csv(buf, index=False)
    return buf.getvalue()
