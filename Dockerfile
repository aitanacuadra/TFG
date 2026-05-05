# Usamos una imagen oficial y ligera de Python
FROM python:3.10-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos primero el archivo de dependencias (buena práctica para aprovechar la caché de Docker)
COPY requirements.txt .

# Instalamos las librerías necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código fuente del backend
COPY src/ ./src/

# Exponemos el puerto en el que correrá FastAPI (por defecto suele ser 8000)
EXPOSE 8000

# Comando para arrancar la aplicación
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]