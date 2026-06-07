---
sidebar_position: 4
title: Arquitectura
---

# Arquitectura del sistema

El sistema sigue una arquitectura de microservicios orquestada con Docker Compose. Cada responsabilidad está aislada en su propio módulo, lo que facilita el mantenimiento y la escalabilidad: cambiar el modelo de lenguaje, la base de datos vectorial o el evaluador no requiere tocar el resto del pipeline.

---

## Infraestructura y contenedores Docker

El entorno de despliegue define tres servicios Docker. Para generar metadatos solo son necesarios `api` y `qdrant`; el contenedor `docs` es opcional y sirve únicamente para el desarrollo local de la documentación.

![Diagrama de infraestructura y contenedores Docker](/img/figura6-docker.png)

*Figura 6: Diagrama de infraestructura y contenedores Docker.*

- **`api`** — Núcleo de la aplicación. Ejecuta el servidor FastAPI, coordina todo el pipeline y devuelve los metadatos DCAT-AP junto con su evaluación de calidad. Se comunica con Qdrant (dentro de Docker) y con Ollama (en el host) via `host.docker.internal`.
- **`qdrant`** — Base de datos vectorial que almacena dos colecciones: los fragmentos de la normativa DCAT-AP (para el RAG) y los metadatos generados de cada dataset procesado. Los datos persisten en el volumen `/qdrant_data`.
- **`docs`** — Web de documentación construida con Docusaurus. En local se levanta en el puerto 3000; en producción se despliega como sitio estático en GitHub Pages.
- **Ollama (HOST)** — Se ejecuta fuera de Docker porque necesita acceso directo al procesador. Sirve tanto el modelo de generación (`gemma3:1b`) como el de embeddings (`nomic-embed-text`).

---

## Capas lógicas

El pipeline se organiza en cinco capas que transforman progresivamente el archivo de entrada en metadatos estructurados y evaluados:

![Flujo por capas](/img/figura7-capas.png)

*Figura 7: Flujo por capas.*

---

## Estructura del proyecto

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
        ├── ai_service.py        # LLM + embeddings + RAG (Ollama/LangChain)
        ├── metadata_service.py  # Construcción del JSON-LD DCAT-AP
        ├── judge_service.py     # Evaluación de calidad (Gemini)
        ├── qdrant_service.py    # Inserción de vectores en Qdrant
        └── rag_service.py       # Inicialización de la base de datos RAG
```

---

## Servicios

### `file_service` — Ingesta y perfilado

Lee el archivo subido y determina automáticamente si es un CSV o un JSON, qué codificación usa y, en el caso del CSV, qué separador de columnas tiene. Para evitar problemas con archivos grandes, el análisis se limita siempre a las primeras 100 filas.

El resultado es un **perfil estructural** con los nombres de columna, los tipos de datos traducidos al estándar XSD, los valores nulos por columna y ejemplos representativos. Junto con una muestra de las primeras filas, es lo que recibe el modelo de lenguaje en el siguiente paso.

---

### `ai_service` — RAG y generación de metadatos

Coordina el proceso completo de generación usando LangChain. Cuando llega una petición, consulta la base de conocimiento normativa en Qdrant para recuperar los fragmentos más relevantes de la especificación DCAT-AP. Con ese contexto, el perfil del dataset y la muestra de datos, construye el prompt y llama al modelo local (Ollama · Gemma 3) para que genere los metadatos en formato JSON.

Si Qdrant no está disponible al arrancar, el servidor continúa funcionando pero avisa del problema cuando llega la primera petición.

---

### `metadata_service` — Construcción DCAT-AP 3.0

Toma la respuesta JSON del modelo y la convierte en un documento JSON-LD válido conforme a DCAT-AP 3.0. Completa los campos obligatorios (`dct:title`, `dct:description`, `dcat:keyword`, `dcat:theme`), añade la información de la distribución del archivo e incluye una entrada por cada columna del dataset con su tipo de dato. También deja constancia de que los metadatos fueron generados por IA.

---

### `judge_service` — Evaluación de calidad

Envía los metadatos generados a Google Gemini, que actúa como juez externo y los puntúa según la metodología MQA de data.europa.eu: hasta 405 puntos repartidos en cinco dimensiones, junto con recomendaciones de mejora. Se usa un modelo diferente al de la generación para que la evaluación sea imparcial.

Si no hay clave de Gemini configurada, devuelve una evaluación vacía y el proceso continúa sin interrumpirse.

---

### `qdrant_service` — Almacenamiento vectorial

Guarda los metadatos generados en Qdrant como un vector numérico, lo que permite búsquedas semánticas posteriores. El vector se obtiene a partir del título, la descripción y las palabras clave del dataset.

---

### `rag_service` — Inicialización de la base normativa

Prepara la base de conocimiento que usa el sistema para generar metadatos correctos. Carga la especificación oficial DCAT-AP, la divide en fragmentos pequeños, convierte cada fragmento en un vector numérico y los almacena en Qdrant. Este proceso solo hay que ejecutarlo una vez: a partir de ahí, cada vez que llega una petición el sistema puede consultar esa base para recuperar el contexto normativo relevante.
