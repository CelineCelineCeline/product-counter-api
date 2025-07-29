from ultralytics import YOLO

model = YOLO("yolo11n-obb.pt")

results = model.train(
    data="yolo_dataset/config.yaml",
    epochs=100,
    patience=25,
    imgsz=1024,
    project='./results',
    device=-1,
    plots=True
)

