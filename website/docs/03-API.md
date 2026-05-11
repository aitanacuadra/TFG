---
sidebar_position: 3
title: Uso de la API
---

# Uso de la API

La documentación interactiva (Swagger UI) está disponible en http://localhost:8000/docs una vez levantados los servicios.

---

## Autenticación

Todas las peticiones requieren la cabecera `X-API-Key` con el valor de `API_KEY` definido en tu `.env`.

---

## Endpoint principal

### `POST /api/v1/process`

Sube un archivo CSV o JSON para generar sus metadatos DCAT-AP 3.0, vectorizarlos en Qdrant y evaluarlos con el modelo juez.

**Parámetro requerido:** archivo `.csv` o `.json` (máximo 10 MB).

**Ejemplo con `curl`:**

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -H "X-API-Key: tu-api-key-secreta" \
  -F "file=@dataset.csv"
```

**Respuesta:**

```json
{
  "message": "Procesado y vectorizado con éxito",
  "output_filename": "dataset.csv_meta.json",
  "metadata": {
    "@type": "dcat:Dataset",
    "dct:title": "...",
    "dct:description": "...",
    "dcat:keyword": ["...", "..."],
    "dcat:distribution": [ { "..." } ],
    "schema:variableMeasured": [ { "..." } ]
  },
  "evaluation": {
    "score_global": 285,
    "categoria_mqa": "Buena (221-350)",
    "analisis_detallado": {
      "findability": "...",
      "accessibility": "...",
      "interoperability": "...",
      "reusability": "...",
      "contextuality": "..."
    },
    "mejoras_prioritarias": ["..."]
  }
}
```

---

## Puntuación MQA

La evaluación sigue la metodología [MQA de data.europa.eu](https://data.europa.eu/mqa/methodology) con cinco dimensiones:

| Dimensión | Máx. puntos | Qué evalúa |
|-----------|-------------|------------|
| Findability | 100 | Palabras clave, temas, localización y temporalidad |
| Accessibility | 100 | Presencia y accesibilidad de `accessURL` y `downloadURL` |
| Interoperability | 130 | Formatos abiertos y conformidad con DCAT-AP 3.0 |
| Reusability | 75 | Licencia, contacto, editor y restricciones de acceso |
| Contextuality | 20 | Derechos, tamaño del archivo y fechas |

| Categoría | Rango |
|-----------|-------|
| Excelente | 351 – 425 |
| Buena | 221 – 350 |
| Suficiente | 121 – 220 |
| Mala | 0 – 120 |
