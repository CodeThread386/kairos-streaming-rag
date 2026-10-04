FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    KAIROS_INDEX_CACHE=/app/.kairos_cache KAIROS_TRACE_DIR=/app/traces
WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY kairos ./kairos
COPY web ./web
COPY demo ./demo
COPY corpus ./corpus
COPY bench ./bench
COPY docs ./docs
COPY tests ./tests
COPY examples ./examples

# Build both indexes at image-build time (CPU only, no network needed: the embedding weights ship in the wheel)
RUN python -m kairos index --corpus corpus/tessera && python -m kairos index --corpus corpus/handbook18f

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["python", "-m", "kairos", "serve", "--host", "0.0.0.0", "--port", "8000"]
