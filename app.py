from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Fashion-MNIST API",
    description="API de classification des vêtements Fashion-MNIST",
    version="1.0",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "fashion_mnist.keras"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

class ImageInput(BaseModel):
    pixels: list[float]

print("Chemin du modèle :", MODEL_PATH)
print("Fichier trouvé :", MODEL_PATH.exists())
model = tf.keras.models.load_model(MODEL_PATH)
print("Modèle chargé avec succès")
print("Forme d'entrée :", model.input_shape)
print("Forme de sortie :", model.output_shape)

@app.get("/")
def home():
    return {"message": "Bienvenue dans Fashion-MNIST API"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": MODEL_PATH.name,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
    }

@app.post("/predict")
def predict(image: ImageInput):
    if len(image.pixels) != 784:
        raise HTTPException(
            status_code=400,
            detail="L'entrée doit contenir exactement 784 pixels.",
        )

    pixels = np.asarray(image.pixels, dtype=np.float32)
    if not np.all(np.isfinite(pixels)):
        raise HTTPException(status_code=400, detail="Les pixels doivent être des nombres finis.")
    if np.any(pixels < 0) or np.any(pixels > 255):
        raise HTTPException(status_code=400, detail="Les pixels doivent être compris entre 0 et 255.")

    batch = pixels.reshape(1, 28, 28)
    if np.max(batch) > 1.0:
        batch = batch / 255.0

    start = time.perf_counter()
    logits = model(tf.convert_to_tensor(batch), training=False)
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
    latency_ms = (time.perf_counter() - start) * 1000

    class_id = int(np.argmax(probabilities))
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": round(float(probabilities[class_id]), 4),
        "latency_ms": round(latency_ms, 2),
    }