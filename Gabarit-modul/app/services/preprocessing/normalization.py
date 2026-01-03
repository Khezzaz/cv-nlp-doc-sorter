from typing import List, Dict

def normalize_blocks(blocks: List[Dict], image_w: int, image_h: int) -> List[Dict]:
    """
    Convert pixel bboxes to normalized [0,1] coordinates and compute normalized area and center.
    Input block: {"bbox":[x_min, y_min, x_max, y_max], "confidence": float}
    """
    normalized = []
    if image_w <= 0 or image_h <= 0:
        return normalized
    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        # clamp
        x0 = max(0, min(x0, image_w))
        x1 = max(0, min(x1, image_w))
        y0 = max(0, min(y0, image_h))
        y1 = max(0, min(y1, image_h))
        w = max(0, x1 - x0)
        h = max(0, y1 - y0)
        if w == 0 or h == 0:
            continue
        nx0 = x0 / image_w
        ny0 = y0 / image_h
        nx1 = x1 / image_w
        ny1 = y1 / image_h
        area = w * h / (image_w * image_h)
        cx = (nx0 + nx1) / 2.0
        cy = (ny0 + ny1) / 2.0
        normalized.append({
            "bbox": [nx0, ny0, nx1, ny1],
            "area": area,
            "center": [cx, cy],
            "confidence": b.get("confidence", 1.0)
        })
    return normalized
