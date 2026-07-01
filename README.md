# Whiskey-Bottle-Analyser-Backend
Standalone Python service for YOLO detection, object tracking, and label OCR. The installable frontend sends JPEG frames to this service and receives normalized JSON detections.

## Run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
MODEL_PATH=""MODEL_PATH python run.py
```

The API starts on `0.0.0.0:8765`.

- `GET /health` — readiness, model path, and OCR backend
- `POST /infer` — multipart JPEG/PNG in the `frame` field

macOS uses Apple Vision automatically. Other systems fall back to Tesseract. For mobile clients, configure `SSL_CERTFILE` and `SSL_KEYFILE` with a certificate trusted by the phone.