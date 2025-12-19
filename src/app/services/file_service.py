import io
import json
import pandas as pd
from typing import Tuple, Dict
from fastapi import HTTPException

# En src/app/services/file_service.py

def sniff_dataframe(file_bytes: bytes, content_type: str) -> Tuple[pd.DataFrame, str]:

    try:
      
        try:
            text = file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1").strip()

    
        if text and (text.startswith("{") or text.startswith("[")):
            try:
                obj = json.loads(text)
                # Normalizamos según sea lista o dict
                if isinstance(obj, list):
                    df = pd.DataFrame(obj)
                elif isinstance(obj, dict):
                    df = pd.json_normalize(obj)
                else:
                    raise ValueError("JSON válido pero estructura no tabular")
                
                print("✅ Archivo detectado como JSON por su contenido.")
                return df, "json"
            except json.JSONDecodeError:
                pass # Parecía JSON pero no lo era, seguimos
    except Exception:
        pass # Falló algo en la lectura, seguimos a CSV

    # ---------------------------------------------------------
    # 2. INTENTO DE CSV (Estrategia Fuerza Bruta)
    # ---------------------------------------------------------
    try:
        encodings_to_try = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        separators_to_try = [",", ";", "\t", "|"]
        
        last_exception = None

        for encoding in encodings_to_try:
            for sep in separators_to_try:
                try:
                    df = pd.read_csv(
                        io.BytesIO(file_bytes), 
                        encoding=encoding, 
                        sep=sep,
                        on_bad_lines='skip'
                    )
                    if df.shape[1] > 1:
                        print(f"✅ CSV detectado: {encoding} | sep: '{sep}'")
                        return df, "csv"
                except Exception as e:
                    last_exception = e
                    continue

        # Último intento permisivo (motor python)
        try:
             df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1", sep=None, engine="python")
             return df, "csv"
        except:
            pass

        raise last_exception or ValueError("No se pudo leer ni como JSON ni como CSV.")

    except Exception as e:
        # Hacemos el error más legible para el usuario
        raise HTTPException(
            status_code=400, 
            detail=f"Error al procesar archivo. Si es un JSON asegúrate de que es válido. Error técnico: {str(e)}"
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
        # ARREGLO: Usamos dropna() para obtener un ejemplo válido de forma segura
        # Esto evita el error "truth value of an array is ambiguous" si la celda contiene una lista
        try:
            valid_series = df[c].dropna()
            if not valid_series.empty:
                # Cogemos el primer valor que no sea nulo
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