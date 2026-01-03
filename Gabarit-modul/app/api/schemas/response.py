from pydantic import BaseModel
from typing import Dict

class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    scores: Dict[str, float]
