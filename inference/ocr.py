from __future__ import annotations

import logging
import platform
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

from .config import Settings


logger = logging.getLogger(__name__)


SWIFT_OCR = r"""
import Vision
import Foundation
import ImageIO

guard CommandLine.arguments.count > 1 else { exit(1) }
let url = URL(fileURLWithPath: CommandLine.arguments[1])
guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { exit(0) }
var recognized = [String]()
let request = VNRecognizeTextRequest { request, _ in
  guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
  recognized = observations.compactMap { $0.topCandidates(1).first?.string }
}
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
try? VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
print(recognized.joined(separator: " "))
"""


class OcrEngine:
    def __init__(self, config: Settings):
        self.config = config
        self.swift_path = Path(tempfile.gettempdir()) / "edge_ai_vision_ocr.swift"
        self.swift_path.write_text(SWIFT_OCR)
        self.backend = "apple-vision" if platform.system() == "Darwin" else "tesseract"
        logger.info("OCR engine initialized backend=%s", self.backend)

    def recognize(self, crop: np.ndarray) -> str:
        if crop is None or crop.size == 0:
            logger.debug("OCR received empty crop")
            return ""
        height, width = crop.shape[:2]
        logger.debug("Recognizing OCR for crop size=%sx%s backend=%s", width, height, self.backend)
        if width < 300:
            scale = max(2, 300 // max(width, 1) + 1)
            crop = cv2.resize(crop, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
            logger.debug("Resized OCR crop to %sx%s", crop.shape[1], crop.shape[0])
        if self.backend == "apple-vision":
            text = self._apple_vision(crop)
            if text:
                logger.debug("Apple Vision OCR returned text=%s", text)
                return text
        text = self._tesseract(crop)
        logger.debug("Tesseract OCR returned text=%s", text)
        return text

    def _apple_vision(self, crop: np.ndarray) -> str:
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                temp_path = temp.name
            cv2.imwrite(temp_path, crop)
            result = subprocess.run(
                ["swift", str(self.swift_path), temp_path],
                capture_output=True, text=True, timeout=5, check=False,
            )
            logger.debug("Apple Vision subprocess exited=%s stdout=%s stderr=%s", result.returncode, result.stdout.strip(), result.stderr.strip())
            return result.stdout.strip().upper()
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            logger.warning("Apple Vision OCR failed: %s", error)
            return ""
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    @staticmethod
    def _tesseract(crop: np.ndarray) -> str:
        try:
            import pytesseract
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            return pytesseract.image_to_string(gray).upper().strip()
        except Exception as error:
            logger.warning("Tesseract OCR failed: %s", error)
            return ""

    def match_brand(self, text: str) -> tuple[str | None, int]:
        if not text:
            return None, 0
        scored = [
            (brand, int(max(fuzz.partial_ratio(brand, text), fuzz.token_set_ratio(brand, text))))
            for brand in self.config.target_brands
        ]
        brand, score = max(scored, key=lambda item: item[1])
        return (brand, score) if score >= self.config.fuzzy_threshold else (None, score)