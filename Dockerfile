FROM python:3.11-slim

# Bibliotecas do sistema que o OpenCV (usado pelo YOLO) precisa
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
WORKDIR /app

# PyTorch só para CPU (bem mais leve) e depois as dependências do projeto
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Código da API e modelo treinado
COPY main.py .
COPY models/best.pt models/best.pt

# O Cloud Run informa a porta pela variável PORT (padrão 8080)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
