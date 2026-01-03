from pydantic import BaseModel, conlist, confloat
from typing import List, Optional

class Block(BaseModel):
    bbox: conlist(int, min_items=4, max_items=4) 
    confidence: Optional[confloat(ge=0.0, le=1.0)] = 1.0

class OCRInput(BaseModel):
    image_width: int
    image_height: int
    blocks: Optional[List[Block]] = []
