from pydantic import BaseSettings

class Settings(BaseSettings):
    templates_dir: str = "templates"
    min_block_area: float = 0.0005   # normalized area threshold (if using normalized coords)
    min_confidence: float = 0.4
    iou_threshold: float = 0.3
    # weighting for future fusion (not used by this service but convenient)
    weight_template: float = 0.2
    weight_cnn: float = 0.4
    weight_nlp: float = 0.4

    class Config:
        env_prefix = "TC_"

SETTINGS = Settings()
