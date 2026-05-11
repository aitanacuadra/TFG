---
sidebar_position: 4
title: Arquitectura
---

# Arquitectura del sistema

El proyecto sigue una arquitectura de servicios desacoplados orquestados con Docker Compose. Cada responsabilidad está aislada en su propio módulo dentro de `src/app/`.

```
src/
└── app/
    ├── main.py              # Punto de entrada FastAPI
    ├── schemas.py           # Modelos Pydantic de entrada/salida
    ├── core/
    │   └── config.py        # Variables de entorno (Pydantic Settings)
    ├── api/
    │   ├── deps.py          # Autenticación por API Key
    │   └── v1/
    │       └── endpoints.py # Rutas de la API
    ├── models/
    │   └── run.py           # Modelos SQLModel (auditoría SQL)
    ├── db/
    │   └── session.py       # Conexión y sesión SQLite
    └── services/
        ├── file_service.py      # Ingesta y profiling de archivos
        ├── ai_service.py        # LLM + embeddings + RAG (Ollama)
        ├── metadata_service.py  # Construcción del JSON-LD DCAT-AP
        ├── judge_service.py     # Evaluación de calidad (Gemini)
        └── qdrant_service.py    # Inserción de vectores en Qdrant
```

---

## Servicios

### `file_service` — Ingesta y profiling

Lee el archivo subido (CSV o JSON), detecta automáticamente codificación y separadores, y genera un perfil estructural del dataset:

- Número de filas y columnas
- Tipos de datos por columna
- Conteo de valores nulos
- Ejemplos de valores reales

Para archivos grandes, aplica una poda limitando el análisis a las primeras 100 filas, evitando problemas de memoria.

---

### `ai_service` — RAG y generación de metadatos

Implementa el pipeline RAG completo:

1. Conecta con Qdrant y carga el retriever sobre la colección `dcat_metadata` (normativa DCAT-AP previamente ingestada).
2. Lanza una consulta semántica para recuperar los fragmentos normativos más relevantes.
3. Construye el prompt combinando el contexto normativo, el perfil del dataset y una muestra de datos.
4. Invoca el LLM local (Ollama · `gemma3:1b`) para generar el JSON de metadatos.
5. Expone también `generate_embeddings()` para vectorizar los metadatos generados antes de insertarlos en Qdrant.

---

### `metadata_service` — Construcción DCAT-AP 3.0

Toma la salida cruda del LLM y la alinea con el esquema DCAT-AP 3.0, produciendo un JSON-LD válido con:

- Campos obligatorios: `dct:title`, `dct:description`, `dcat:keyword`, `dcat:theme`
- Distribución: formato, mediaType, byteSize, downloadURL
- Variables medidas: tipo XSD inferido por columna (`xsd:integer`, `xsd:decimal`, `xsd:dateTime`…)
- Proveniencia: indica que los metadatos fueron generados por IA

---

### `judge_service` — Evaluación de calidad (LLM-as-a-judge)

Usa Google Gemini como juez externo para evaluar los metadatos generados siguiendo la metodología MQA de data.europa.eu. Devuelve una puntuación global y un análisis por dimensión con recomendaciones de mejora.

---

### `qdrant_service` — Almacenamiento vectorial

Inserta los metadatos en Qdrant como un punto vectorial:

- **Vector:** embedding del título + descripción + palabras clave (modelo `nomic-embed-text`).
- **Payload:** el JSON-LD DCAT-AP completo.
- **ID:** UUID determinista derivado del ID de auditoría en SQLite, manteniendo la trazabilidad entre ambas bases de datos.

---

## Almacenamiento dual

El sistema persiste la información en dos capas complementarias:

| Capa | Tecnología | Propósito |
|------|-----------|-----------|
| Relacional | SQLite (SQLModel) | Auditoría: quién subió qué, cuándo y con qué resultado |
| Vectorial | Qdrant | Búsqueda semántica sobre los metadatos generados |

Ambos registros se enlazan mediante el mismo ID, permitiendo trazabilidad completa de cada ejecución.
