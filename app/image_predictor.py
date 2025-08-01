from ultralytics import YOLO
from PIL import Image
from typing import Dict
import io


class DetectionModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def predict(self, image_bytes: bytes) -> Dict[str, int]:
        image = Image.open(io.BytesIO(image_bytes))

        results = self.model(image)

        counts = {}
        for result in results:
            for box in result.obb:
                class_id = int(box.cls)
                class_name = self.class_names[class_id]
                counts[class_name] = counts.get(class_name, 0) + 1

        return counts
