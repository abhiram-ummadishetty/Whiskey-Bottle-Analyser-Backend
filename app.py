import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference.config import settings
from inference.logging_config import configure_logging
from inference.pipeline import BottlePipeline


configure_logging()
logger = logging.getLogger(__name__)

pipeline: BottlePipeline | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipeline
    pipeline = BottlePipeline(settings)
    yield
    pipeline = None


app = FastAPI(title="Backend Processor", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "ready": pipeline is not None,
        "model": str(settings.model_path),
        "ocr_backend": pipeline.ocr.backend if pipeline else "unavailable",
    }


@app.post("/infer")
async def infer(frame: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if frame.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    try:
        payload = await frame.read()
        if len(payload) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Frame exceeds the 10 MB limit")
        return await pipeline.infer(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error