# Diseño e implementación de un sistema de generación automatizada de metadatos DCAT-AP basado en Modelos de Lenguaje de Gran Escala (LLMs)
**Trabajo de Fin de Grado – Ingeniería de Tecnologías y Servicios de Telecomunicación (UPM)**

Sistema que automatiza la catalogación de archivos CSV y JSON generando metadatos conformes al estándar europeo DCAT-AP 3.0. Recibe un archivo, analiza su estructura con Pandas, recupera contexto normativo de la especificación DCAT-AP mediante RAG, y usa un modelo de lenguaje para generar los metadatos. La calidad del resultado se evalúa automáticamente con la metodología MQA de data.europa.eu.

> **Documentación Completa:** Puedes consultar la guía de instalación y detalles más completos en la [Web de Documentación del Proyecto](https://aitanacuadra.github.io/TFG/).

---

![Flujo por capas del sistema](website/static/img/figura7-capas.png)

---

## Tecnologías utilizadas

| Tipo | Herramienta |
|------|-------------|
| Lenguaje | Python 3.13 |
| API Framework | FastAPI |
| Procesamiento de datos | Pandas |
| Orquestación de IA | LangChain |
| Base de datos vectorial | Qdrant |
| LLM generación de metadatos | Ollama · `gemma3:1b` |
| LLM evaluador (juez) | Google Gemini |
| Base de datos relacional | SQLite (SQLModel) |
| Despliegue | Docker · Docker Compose |

---

## Instalación

### Prerrequisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución.
- [Ollama](https://ollama.com/download) instalado en el host

Descarga los modelos necesarios:

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```
> Son los modelos predeterminados. Puedes usar cualquier otro modelo disponible en [ollama.com/library](https://ollama.com/library) y cambia `OLLAMA_MODEL` y `EMBEDDINGS_MODEL` en tu `.env`.

### 1. Clonar el repositorio

```bash
git clone https://github.com/aitanacuadra/TFG
cd TFG
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` y sustituye los valores marcados con `<...>`

### 3. Levantar los servicios

```bash
docker compose up --build
```

Esto arranca tres servicios:
- **API** → http://localhost:8000
- **Documentación** (Docusaurus) → http://localhost:3000/TFG
- **Qdrant** → http://localhost:6333

---

## Uso de la API

Disponible en http://localhost:8000/docs.

### Endpoint principal

**`POST /api/v1/process`**

Sube un archivo CSV o JSON para generar sus metadatos DCAT-AP 3.0, vectorizarlos en Qdrant y evaluarlos con el modelo juez.

**Autenticación:** cabecera `X-API-Key` con el valor de `API_KEY`.

### Parámetro requerido
Archivo:   
.json o .csv

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
    ...
  },
  "evaluation": {
    "score_global": 285,
    "categoria_mqa": "Buena (221-350)",
    "analisis_detallado": { ... },
    "mejoras_prioritarias": ["..."]
  }
}
```

---

## Autora

**Aitana Cuadra** — [GitHub](https://github.com/aitanacuadra)
