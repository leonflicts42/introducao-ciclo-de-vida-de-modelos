# Dockerfile específico da Aula 4 (Deployment)
# Construir usando o contexto da raiz do repositório:
#   docker build -t heart-api:latest -f aula_04_implantacao/Dockerfile .
# Isso garante acesso a requirements.txt, models/ e demais pastas.

FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primeiro (cache layer)
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da API Flask (a partir da raiz do repo no contexto de build)
COPY aula_04_implantacao/flask-app ./aula_04_implantacao/flask-app
# Copiar modelos necessários para inferência
COPY models ./models

ENV MODEL_PATH=/app/models/best_random_forest.joblib
WORKDIR /app/aula_04_implantacao/flask-app

EXPOSE 5000

# Entrada padrão (para produção pode-se trocar por gunicorn)
CMD ["python", "app.py"]
