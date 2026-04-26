# Automatización de la generación de metadatos DCAT-AP mediante Modelos de Lenguaje de Gran Escala
**Trabajo de Fin de Grado – Ingeniería de Tecnologías y Servicios de Telecomunicación (UPM)**

El presente proyecto desarrolla una solución orientada a la mejora de la gestión y calidad de los metadatos. El objetivo principal consiste en automatizar la generación de metadatos conforme al estándar europeo DCAT-AP, transformando datos brutos en recursos estructurados, localizables y reutilizables. Para ello, se ha desarrollado una arquitectura basada en un sistema RAG que integra FastAPI, LangChain y Ollama. El sistema realiza un análisis de archivos en formato CSV y JSON para identificar su contenido y generar automáticamente metadatos alineados con el esquema DCAT-AP. 


> **Documentación Completa:** Puedes consultar la guía de instalación y detalles mas completos de mi proyecto en la [Web de Documentación del Proyecto](https://aitanacuadra.github.io/TFG/).





```mermaid
graph LR
    %% Definición de estilos
    classDef input fill:#E3F2FD,stroke:#1565C0,stroke-width:2px;
    classDef api fill:#4A148C,color:#fff,stroke:#311B92,stroke-width:2px;
    classDef ai fill:#FF8F00,color:#fff,stroke:#EF6C00,stroke-width:2px;
    classDef audit fill:#2E7D32,color:#fff,stroke:#1B5E20,stroke-width:2px;
    classDef process fill:#F5F5F5,stroke:#616161,stroke-dasharray: 5 5;

    %% Flujo
    User([Usuario]) --> API[FastAPI]
    
    subgraph Pipeline ["Pipeline de Procesamiento de Datos"]
        API --> Profiling[Análisis del archivo]
        Profiling --> RAG[Extracción Semántica]
        RAG --> DCAT[Generación DCAT-AP]
        DCAT --> Judge[Evaluación de Calidad MQA]
    end
    
    Judge --> API
    API --> Output[(Respuesta Final)]

    %% Aplicación de estilos
    class User,Output input;
    class API api;
    class RAG ai;
    class Judge audit;
    class Profiling,DCAT process;
```
---

## Tecnologías utilizadas

| Tipo | Herramienta |
|------|-------------|
| Lenguaje | Python 3.13 |
| API Framework | FastAPI |
| Procesamiento de datos | Pandas |
| Orquestación de IA | LangChain |
| Base de datos vectorial para RAG | FAISS |
| LLM extracción metadatos | Ollama(gemma3:1b) |
| LLM as a judge | gemini-3-flash-preview |


--- 
## Instalación

### 1. Prerrequisitos

Instalar **Ollama**:  https://ollama.com/download

Descargar el modelo utilizado:

```bash
ollama pull gemma3:1b
```


### 2. Clonar el repositorio
```bash
git clone https://github.com/aitanacuadra/TFG
cd TFG
```

### 3. Crear entorno virtual 
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows
```

### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 5. Configurar variables de entorno

Crear un archivo .env:
```bash
BASE_DATASET_URL=https://example.org/datasets/
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:1b

```

### 6. Ejecutar la API
```bash
PYTHONPATH=src uvicorn app.main:app --reload
```
La API estará disponible en: http://localhost:8000



## Uso de la API
### Endpoint principal
POST /procesar

### Parámetro requerido
archivo .json o .csv

# Autora: Aitana Cuadra






