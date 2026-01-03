from typing import List, Dict, Any
from .iou import compute_iou
from app.core.config import SETTINGS
from app.utils.geometry import bbox_center_size

def match_templates_scores(ocr_blocks: List[Dict], templates: Dict[str, Any], iou_threshold: float = 0.3) -> Dict[str, float]:
    """
    For each template, compute a normalized similarity score based on IoU matching.
    templates: mapping name -> template structure
      expected template format:
        {
          "zones": [
            {"name": "logo", "bbox": [x0,y0,x1,y1]},
            ...
          ]
        }
    """
    scores = {}
    # prepare doc boxes list
    doc_boxes = [b["bbox"] for b in ocr_blocks]

    for name, tpl in templates.items():
        zones = tpl.get("zones", [])
        if not zones:
            scores[name] = 0.0
            continue
        total = 0.0
        for zone in zones:
            zone_bbox = zone.get("bbox")
            best_iou = 0.0
            for db in doc_boxes:
                iou = compute_iou(zone_bbox, db)
                if iou > best_iou:
                    best_iou = iou
            # enforce threshold: if best_iou < threshold treat as 0
            if best_iou < iou_threshold:
                best_iou = 0.0
            total += best_iou
        score = total / max(1, len(zones))
        scores[name] = float(score)
    # normalize scores to [0,1] (they already are)
    return scores
