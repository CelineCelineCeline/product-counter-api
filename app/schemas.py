from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    items: Dict[str, int]
