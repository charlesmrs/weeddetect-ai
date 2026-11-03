import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

# Carrega o modelo uma única vez, quando a API inicia
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

# Detecções com confiança abaixo deste valor são descartadas
CONFIANCA_MINIMA = 0.25

# Tipos de arquivo aceitos no upload
TIPOS_ACEITOS = {"image/jpeg", "image/png"}

app = FastAPI(
    title="WeedDetect AI",
    description="API de detecção de plantas daninhas em lavouras de algodão com YOLO11s",
    version="1.0.0",
)


def nome_da_classe(classe_id: int) -> str:
    """Devolve o nome da classe sem o prefixo 'weed- '."""
    return model.names[classe_id].replace("weed- ", "")


@app.get("/")
def status():
    """Verifica se a API está no ar e lista as classes do modelo."""
    classes = [nome_da_classe(i) for i in model.names]
    return {
        "status": "online",
        "modelo": "YOLO11s",
        "total_classes": len(classes),
        "classes": classes,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Recebe uma imagem e devolve as plantas daninhas detectadas."""
    # 1. Confere se o arquivo é uma imagem JPG ou PNG
    if file.content_type not in TIPOS_ACEITOS:
        raise HTTPException(
            status_code=400,
            detail="Formato inválido. Envie uma imagem JPG ou PNG.",
        )

    # 2. Lê o arquivo e abre como imagem
    conteudo = await file.read()
    try:
        imagem = Image.open(io.BytesIO(conteudo)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível abrir a imagem enviada.",
        )

    # 3. Roda o modelo YOLO na imagem
    resultado = model.predict(imagem, conf=CONFIANCA_MINIMA, verbose=False)[0]

    # 4. Monta a lista de detecções no formato do enunciado
    deteccoes = []
    for caixa in resultado.boxes:
        x_min, y_min, x_max, y_max = caixa.xyxy[0].tolist()
        deteccoes.append({
            "class_name": nome_da_classe(int(caixa.cls[0])),
            "confidence": round(float(caixa.conf[0]), 2),
            "box": {
                "x_min": int(x_min),
                "y_min": int(y_min),
                "x_max": int(x_max),
                "y_max": int(y_max),
            },
        })

    return {"status": "success", "detections": deteccoes}