from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv(
        "MODEL_PATH",
        "",
    ))
    confidence: float = float(os.getenv("YOLO_CONFIDENCE", "0.5"))
    ocr_confidence: float = float(os.getenv("OCR_CONFIDENCE", "0.75"))
    fuzzy_threshold: int = int(os.getenv("FUZZY_THRESHOLD", "60"))
    ocr_cooldown_seconds: float = float(os.getenv("OCR_COOLDOWN_SECONDS", "1.5"))
    track_ttl_seconds: float = float(os.getenv("TRACK_TTL_SECONDS", "1.0"))
    target_brands: tuple[str, ...] = field(default_factory=lambda: (
        "JIM BEAM", "MAKERS MARK", "MAKER'S MARK", "HIBIKI",
        "SUNTORY", "SUNTORY HIBIKI",
    ))


settings = Settings()