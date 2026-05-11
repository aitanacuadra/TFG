---
sidebar_position: 2
title: Instalación
---

# Instalación

## Prerrequisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución.
- [Ollama](https://ollama.com/download) instalado en el host (corre fuera de Docker para aprovechar la GPU).

Descarga los modelos necesarios:

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

> Son los modelos predeterminados. Puedes usar cualquier otro modelo disponible en [ollama.com/library](https://ollama.com/library) cambiando `OLLAMA_MODEL` y `EMBEDDINGS_MODEL` en tu `.env`.

---

## 1. Clonar el repositorio

```bash
git clone https://github.com/aitanacuadra/TFG
cd TFG
```

## 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` y sustituye los valores marcados con `<...>`. Solo necesitas rellenar dos campos:

| Variable | Descripción |
|----------|-------------|
| `API_KEY` | Clave para autenticar las peticiones a la API (pon la que quieras) |
| `GEMINI_API_KEY` | Clave de Google Gemini · [Obtener aquí](https://aistudio.google.com/app/apikey) |

El resto de variables ya están preconfiguradas para Docker en el `.env.example`.

## 3. Levantar los servicios

```bash
docker compose up --build
```

Esto arranca tres servicios:

| Servicio | URL |
|----------|-----|
| API (Swagger UI) | http://localhost:8000/docs |
| Documentación (Docusaurus) | http://localhost:3000/TFG/ |
| Qdrant (panel de control) | http://localhost:6333/dashboard |
