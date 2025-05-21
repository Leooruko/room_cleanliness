from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io
from typing import List
from tensorflow.keras.models import load_model

app = FastAPI()
model = load_model("room_cleanliness.h5")

def preprocess(file: UploadFile):
    img = Image.open(io.BytesIO(file.file.read()))
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    scores = []
    for file in files:
        try:
            img = preprocess(file)
            pred = model.predict(img)[0][0]
            scores.append(pred)
        except Exception:
            continue
    avg_score = sum(scores) / len(scores) if scores else 0
    verdict = "Clean" if avg_score >= 0.5 else "Messy"
    return {"score": round(avg_score, 2), "verdict": verdict}
