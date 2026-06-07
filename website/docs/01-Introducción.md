---
sidebar_position: 1
slug: /
title: Introducción
---

# Diseño e implementación de un sistema de generación automatizada de metadatos DCAT-AP basado en Modelos de Lenguaje de Gran Escala (LLMs)

**Trabajo de Fin de Grado – Ingeniería de Tecnologías y Servicios de Telecomunicación (UPM)**  
**Aitana Cuadra Cano**

El sistema automatiza la catalogación de archivos CSV y JSON generando metadatos conformes al estándar europeo DCAT-AP 3.0. Se basa en una arquitectura RAG que proporciona al modelo de lenguaje el contexto normativo necesario —extraído de la especificación oficial DCAT-AP— para que genere metadatos estructurados alineados con el estándar. La calidad del resultado se evalúa automáticamente con la metodología MQA de data.europa.eu.

---

![Arquitectura general del sistema y flujo de datos](/img/figura5-arquitectura.png)

*Figura 5: Arquitectura general del sistema y flujo de datos.*

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

## Capas del pipeline

1. **Ingesta y validación** — FastAPI recibe la petición. Pydantic verifica la API Key en la cabecera `X-API-Key`, que el archivo no supera los 10 MB y que no está vacío. Si pasa todas las comprobaciones, se registra la ejecución en SQLite con estado `started`.

2. **Procesamiento y perfilado** — Pandas detecta el formato del archivo: para JSON comprueba los primeros bytes y aplana estructuras anidadas con `pd.json_normalize`; para CSV prueba combinaciones de codificación y separador sobre los primeros 10 KB. Después analiza las primeras 100 filas y genera un perfil estructural con los nombres de columna, tipos de datos traducidos a XSD y ejemplos representativos.

3. **Recuperación RAG** — El sistema convierte una consulta fija sobre DCAT-AP en un embedding con `nomic-embed-text` (via Ollama en el HOST) y lo compara por similitud de coseno contra los fragmentos de la normativa indexados en Qdrant. Recupera los 5 fragmentos más relevantes, que se inyectan en el prompt como contexto normativo.

4. **Generación y alineación DCAT-AP** — LangChain construye el prompt combinando el contexto normativo del RAG, el perfil del dataset y la muestra de datos. El LLM local (Ollama · Gemma 3 de 1B parámetros, temperatura 0) genera un JSON con título, descripción, palabras clave y tema. El servicio `metadata_service` lo alinea con el esquema DCAT-AP 3.0 produciendo un JSON-LD con `@context`, distribución y variables medidas con sus tipos XSD.

5. **Evaluación y persistencia** — Google Gemini actúa como juez externo y evalúa los metadatos según las 5 dimensiones MQA de data.europa.eu (hasta 405 puntos): Findability, Accessibility, Interoperability, Reusability y Contextuality. Los metadatos se vectorizan e indexan en Qdrant para búsqueda semántica futura. El registro de SQLite se actualiza con estado `completed` y los resultados de la evaluación.
