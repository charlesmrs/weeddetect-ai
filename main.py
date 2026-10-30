from fastapi import FastAPI
from ultralytics import YOLO

# Carrega o modelo uma única vez, quando a API inicia
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

app = FastAPI(
    title="WeedDetect AI",
    description="API de detecção de plantas daninhas em lavouras de algodão com YOLO11s",
    version="1.0.0",
)


@app.get("/")
def status():
    """Verifica se a API está no ar e lista as classes do modelo."""
    classes = [nome.replace("weed- ", "") for nome in model.names.values()]
    return {
        "status": "online",
        "modelo": "YOLO11s",
        "total_classes": len(classes),
        "classes": classes,
    }