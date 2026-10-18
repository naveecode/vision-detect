"""HTTP API: ``POST /detect``, ``GET /health``, ``GET /version``."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from vision_detect import __version__
from vision_detect.api.inference import OnnxPersonDetector, decode_image_bytes
from vision_detect.constants import MAX_UPLOAD_BYTES
from vision_detect.paths import model_path


class ImageSize(BaseModel):
    width: int
    height: int


class DetectResponse(BaseModel):
    boxes: list[list[float]]
    labels: list[str]
    scores: list[float]
    image_size: ImageSize
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class VersionResponse(BaseModel):
    model_name: str
    model_version: str
    framework: str
    input_size: int
    classes: list[str]
    dataset: str
    nms_iou: float
    sha256: str = ""
    package_version: str = Field(default=__version__)


def create_app(detector: OnnxPersonDetector | None = None, load_default: bool = True) -> FastAPI:
    app = FastAPI(
        title="vision-detect",
        version=__version__,
        description=(
            "Person detection. Upload an image and receive boxes, labels, scores, and latency."
        ),
    )
    if detector is None and load_default and model_path().exists():
        detector = OnnxPersonDetector.from_env()
    app.state.detector = detector

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {
            "service": "vision-detect",
            "health": "/health",
            "version": "/version",
            "detect": "/detect",
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        loaded = app.state.detector is not None
        return HealthResponse(status="ok" if loaded else "no-model", model_loaded=loaded)

    @app.get("/version", response_model=VersionResponse, tags=["meta"])
    def version() -> VersionResponse:
        if app.state.detector is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        payload = app.state.detector.version()
        payload["package_version"] = __version__
        return VersionResponse(**payload)

    @app.post("/detect", response_model=DetectResponse, tags=["detection"])
    async def detect(
        file: UploadFile = File(...),
        score_threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    ) -> DetectResponse:
        if app.state.detector is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            image = decode_image_bytes(payload)
        except ValueError as exc:
            status = 413 if "8 MB" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        return DetectResponse(**app.state.detector.predict(image, score_threshold=score_threshold))

    return app


app = create_app()
