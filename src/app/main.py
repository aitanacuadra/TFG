from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.db.session import init_db
from app.api.v1 import endpoints

app = FastAPI(
    title="API TFG - Metadatos DCAT",
    description="""
Esta API permite analizar archivos **CSV** y **JSON** para generar automáticamente metadatos enriquecidos.

### Enlaces 
* [Ver Documentación Completa](https://aitanacuadra.github.io/TFG/)
* [Ver Código en GitHub](https://github.com/aitanacuadra/TFG)
"""
)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(endpoints.router, prefix="/api/v1")
