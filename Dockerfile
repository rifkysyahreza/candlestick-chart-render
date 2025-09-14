FROM python:3.11-slim

# Keep Python lean and use headless matplotlib
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# System libs required by matplotlib
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libfreetype6 libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Respect platform-provided PORT (Cloud Run sets 8080 by default)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
