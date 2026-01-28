from fastapi import FastAPI

from app.core.config import settings 
from app.db.session import init_db
from app.api.v1 import endpoints
from fastapi.responses import HTMLResponse, RedirectResponse

from fastapi.openapi.docs import get_swagger_ui_html

# 1. Definimos la configuración básica
app = FastAPI(
    title="API TFG - Metadatos DCAT",
    docs_url=None,
    description="""
Esta API permite analizar archivos **CSV** y **JSON** para generar automáticamente metadatos enriquecidos.

### Enlaces 
* [Ver Documentación Completa](https://aitanacuadra.github.io/TFG/)
* [Ver Código en GitHub](https://github.com/aitanacuadra/TFG)

""",
    
    
    
)
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Documentación",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css",
        swagger_favicon_url="https://www.upm.es/favicon.ico",
    )

# ... (Aquí debajo irían tus rutas y middlewares como siempre) ...

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(endpoints.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)