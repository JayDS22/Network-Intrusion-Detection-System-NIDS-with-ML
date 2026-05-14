FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpcap0.8 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-train a tiny ensemble at image-build time so the dashboard works
# the moment the container starts.
RUN python scripts/train_model.py --packets 8000 --epochs 2 --no-tf || true

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
