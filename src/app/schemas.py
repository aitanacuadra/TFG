from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Bloques pequeños (Sub-esquemas) ---

class DCATDistribution(BaseModel):
    type_: str = Field(alias="@type", default="dcat:Distribution")
    id_: str = Field(alias="@id")
    title: str = Field(alias="dct:title")
    description: str = Field(alias="dct:description")
    format_: str = Field(alias="dct:format")
    media_type: str = Field(alias="dcat:mediaType")
    byte_size: int = Field(alias="dcat:byteSize")
    download_url: str = Field(alias="dcat:downloadURL")

    class Config:
        populate_by_name = True

class SchemaPropertyValue(BaseModel):
    type_: str = Field(alias="@type", default="schema:PropertyValue")
    name: str = Field(alias="schema:name")
    value_type: str = Field(alias="schema:valueType")
    description: str = Field(alias="schema:description")

    class Config:
        populate_by_name = True

# --- El objeto principal de Metadatos ---

class DCATDataset(BaseModel):
    context: Dict[str, str] = Field(alias="@context")
    type_: str = Field(alias="@type", default="dcat:Dataset")
    id_: str = Field(alias="@id")
    
    # Campos Dublin Core (dct)
    title: str = Field(alias="dct:title")
    description: str = Field(alias="dct:description")
    identifier: str = Field(alias="dct:identifier")
    language: str = Field(alias="dct:language")
    dct_type: str = Field(alias="dct:type")
    format_: str = Field(alias="dct:format")
    provenance: str = Field(alias="dct:provenance")
    
    # Campos DCAT
    keyword: List[str] = Field(alias="dcat:keyword")
    distribution: List[DCATDistribution] = Field(alias="dcat:distribution")
    
    # Campos Schema.org
    variable_measured: List[SchemaPropertyValue] = Field(alias="schema:variableMeasured")

    class Config:
        populate_by_name = True

# --- La respuesta final de la API ---

class ProcessFileResponse(BaseModel):
    message: str
    output_filename: str
    metadata: DCATDataset