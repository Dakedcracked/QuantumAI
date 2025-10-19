# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

# System deps for psycopg2 and opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project including ML src
COPY . /app

ENV PYTHONPATH="/app"
ENV TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
