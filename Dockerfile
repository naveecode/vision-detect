FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VISION_DETECT_MODEL=/app/web/model/detector.onnx \
    VISION_DETECT_METADATA=/app/web/model/metadata.json

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY vision_detect ./vision_detect
COPY web/model ./web/model

# Serving uses ONNX Runtime. Training dependencies stay out of the image.
RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "vision_detect.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
