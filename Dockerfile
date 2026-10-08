FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/* \
  && useradd --system --create-home --uid 1001 carvo

COPY requirements.txt .
RUN pip install --upgrade pip \
  && pip install -r requirements.txt

COPY app ./app

RUN mkdir -p /app/data/index \
  && chown -R carvo:carvo /app

USER carvo

# Documents the listen port only. Do not publish it on the host; other
# containers reach this service at http://carvo-intelligence:8000.
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
