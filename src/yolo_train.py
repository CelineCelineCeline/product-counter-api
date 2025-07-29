from ultralytics import YOLO

# Load a pretrained YOLO-OBB model
model = YOLO("yolo11n-obb.pt")

# Train the model
results = model.train(
    data="yolo_dataset/config.yaml",
    epochs=100,
    patience=25,
    imgsz=1024,
    project='./results',
    device=-1,
    plots=True
)

