from __future__ import annotations

import logging
from dataclasses import dataclass
import time


logger = logging.getLogger(__name__)


@dataclass
class Track:
    id: int
    box: tuple[float, float, float, float]
    label: str
    confidence: float
    last_seen: float
    last_ocr_at: float = 0.0
    ocr_text: str = ""
    brand: str | None = None
    brand_score: int = 0


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class IoUTracker:
    def __init__(self, ttl_seconds: float, smoothing: float = 0.65):
        self.ttl_seconds = ttl_seconds
        self.smoothing = smoothing
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: list[dict]) -> list[Track]:
        now = time.monotonic()
        unmatched = set(self.tracks)
        logger.debug("Tracking %s detections", len(detections))
        for detection in detections:
            box = tuple(detection["box"])
            candidates = [
                (track_id, iou(box, self.tracks[track_id].box))
                for track_id in unmatched
                if self.tracks[track_id].label == detection["label"]
            ]
            match_id, score = max(candidates, key=lambda item: item[1], default=(0, 0.0))
            if score >= 0.25:
                track = self.tracks[match_id]
                alpha = self.smoothing
                track.box = tuple(alpha * old + (1 - alpha) * new for old, new in zip(track.box, box))
                track.confidence = detection["confidence"]
                track.last_seen = now
                unmatched.remove(match_id)
                detection["track"] = track
                logger.debug("Updated existing track_id=%s score=%s", track.id, score)
            else:
                track = Track(self.next_id, box, detection["label"], detection["confidence"], now)
                self.tracks[self.next_id] = track
                detection["track"] = track
                self.next_id += 1
                logger.debug("Created new track_id=%s label=%s", track.id, detection["label"])
        self.tracks = {
            track_id: track for track_id, track in self.tracks.items()
            if now - track.last_seen <= self.ttl_seconds
        }
        return list(self.tracks.values())