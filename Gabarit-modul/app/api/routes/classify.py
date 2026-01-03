from fastapi import APIRouter, HTTPException
from typing import Dict
from app.api.schemas.ocr_input import OCRInput
from app.api.schemas.response import ClassificationResponse
from app.core.config import SETTINGS
from app.core.logger import get_logger
from app.services.preprocessing.normalization import normalize_blocks
from app.services.preprocessing.filtering import filter_blocks
from app.services.preprocessing.ordering import order_blocks_spatial
from app.services.matching.matcher import match_templates_scores
from pathlib import Path
import json

logger = get_logger(__name__)
router = APIRouter()

TEMPLATES_DIR = Path(SETTINGS.templates_dir)

@router.post("/template", response_model=ClassificationResponse)
def classify_by_template(payload: OCRInput):
    # Validate template dir
    if not TEMPLATES_DIR.exists() or not TEMPLATES_DIR.is_dir():
        logger.error("Templates directory not found: %s", TEMPLATES_DIR)
        raise HTTPException(status_code=500, detail="Templates not available on server.")

    # Load templates
    templates = {}
    for p in TEMPLATES_DIR.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                templates[p.stem] = json.load(f)
        except Exception as e:
            logger.warning("Failed to load template %s: %s", p, e)

    if not templates:
        logger.error("No templates loaded from %s", TEMPLATES_DIR)
        raise HTTPException(status_code=500, detail="No templates loaded on server.")

    # Preprocessing pipeline
    image_w = payload.image_width
    image_h = payload.image_height
    raw_blocks = payload.blocks or []

    normalized = normalize_blocks(raw_blocks, image_w, image_h)
    filtered = filter_blocks(normalized, min_area=SETTINGS.min_block_area, min_confidence=SETTINGS.min_confidence)
    ordered = order_blocks_spatial(filtered)

    # Matching & scoring
    scores = match_templates_scores(ordered, templates, iou_threshold=SETTINGS.iou_threshold)

    # Choose predicted class
    predicted = max(scores.items(), key=lambda kv: kv[1])[0]
    confidence = scores[predicted]

    response = ClassificationResponse(
        predicted_class=predicted,
        confidence=round(confidence, 4),
        scores={k: round(v, 4) for k, v in scores.items()}
    )
    logger.info("Classification result: %s", response.json())
    return response
