from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from .image_predictor import DetectionModel
from .schemas import PredictionResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Product Detection API")

MODEL_PATH = "src/results/train2/weights/best.pt"
model = DetectionModel(MODEL_PATH)

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        counts = model.predict(image_bytes)
        return {"items": counts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
