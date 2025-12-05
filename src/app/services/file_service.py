import io
import json
import pandas as pd
from typing import Tuple, Dict
from fastapi import HTTPException

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
        ex = next((x for x in df[c].tolist() if pd.notna(x)), None)
        profile["examples"][str(c)] = ex
    return profile

def head_as_csv(df: pd.DataFrame, n: int = 5) -> str:
    buf = io.StringIO()
    df.head(n).to_csv(buf, index=False)
    return buf.getvalue()