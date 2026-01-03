from typing import List, Tuple

def bbox_center_size(bbox: List[float]) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    return cx, cy, w, h
