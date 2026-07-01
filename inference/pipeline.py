from __future__ import annotations

import asyncio
import logging
import time

import cv2
import numpy as np
from ultralytics import YOLO

from .config import Settings
from .ocr import OcrEngine
from .tracker import IoUTracker, Track


logger = logging.getLogger(__name__)


class BottlePipeline:
    def __init__(self, config: Settings):
        if not config.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {config.model_path}")
        self.config = config
        logger.info("Loading YOLO model from %s", config.model_path)
        self.model = YOLO(str(config.model_path))
        self.ocr = OcrEngine(config)
        self.tracker = IoUTracker(config.track_ttl_seconds)
        self.lock = asyncio.Lock()
        self.pending_ocr: set[int] = set()
        logger.info("Bottle pipeline initialized with OCR backend=%s", self.ocr.backend)

    async def infer(self, image_bytes: bytes) -> dict:
        logger.debug("Starting inference for image size_bytes=%s", len(image_bytes))
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Failed to decode uploaded image")
            raise ValueError("Invalid image")
        logger.debug("Decoded image shape=%s", image.shape)
        started = time.perf_counter()
        async with self.lock:
            result = await asyncio.to_thread(
                self.model.predict,
                [image],
                conf=self.config.confidence,
                iou=0.45,
                verbose=False,
            )
        detections = []
        names = self.model.names
        for box in result[0].boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0])
            detections.append({
                "box": [x1, y1, x2, y2],
                "label": names[class_id],
                "class_id": class_id,
                "confidence": float(box.conf[0]),
            })
        tracks = self.tracker.update(detections)
        logger.debug("Detected %s objects and produced %s tracks", len(detections), len(tracks))
        for track in tracks:
            if self._should_run_ocr(track):
                self.pending_ocr.add(track.id)
                logger.debug("Scheduling OCR for track_id=%s confidence=%s", track.id, track.confidence)
                asyncio.create_task(self._enrich_track(track, image.copy()))
            else:
                logger.debug("Skipping OCR for track_id=%s reason=confidence_or_cooldown", track.id)
        return {
            "detections": [self._serialize(track) for track in tracks],
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "ocr_backend": self.ocr.backend,
            "width": image.shape[1],
            "height": image.shape[0],
        }

    def _should_run_ocr(self, track: Track) -> bool:
        return (
            track.confidence >= self.config.ocr_confidence
            and track.id not in self.pending_ocr
            and time.monotonic() - track.last_ocr_at >= self.config.ocr_cooldown_seconds
        )

    async def _enrich_track(self, track: Track, image: np.ndarray) -> None:
        try:
            x1, y1, x2, y2 = map(int, track.box)
            crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            logger.debug("Running OCR for track_id=%s crop_shape=%s", track.id, crop.shape)
            text = await asyncio.to_thread(self.ocr.recognize, crop)
            brand, score = self.ocr.match_brand(text)
            current = self.tracker.tracks.get(track.id)
            if current:
                current.ocr_text = text
                current.brand = brand
                current.brand_score = score
                current.last_ocr_at = time.monotonic()
                logger.debug(
                    "OCR completed for track_id=%s text=%s brand=%s score=%s",
                    track.id,
                    text,
                    brand,
                    score,
                )
        finally:
            self.pending_ocr.discard(track.id)

    def _serialize(self, track: Track) -> dict:
        x1, y1, x2, y2 = track.box
        return {
            "track_id": track.id,
            "label": track.label,
            "confidence": track.confidence,
            "x": x1,
            "y": y1,
            "w": max(0, x2 - x1),
            "h": max(0, y2 - y1),
            "ocr_text": track.ocr_text,
            "brand": track.brand,
            "brand_score": track.brand_score,
            "ocr_pending": track.id in self.pending_ocr,
        }