from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from .image_predictor import DetectionModel
from .schemas import PredictionResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Product Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
MODEL_PATH = "src/results/train2/weights/best.pt"
model = DetectionModel(MODEL_PATH)

@app.get("/predict", response_model=PredictionResponse)
def predict():
    image_path = "src/yolo_dataset/train/images/2d97fff7-image00045.jpeg"
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        counts = model.predict(image_bytes)
        return {"items": counts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
