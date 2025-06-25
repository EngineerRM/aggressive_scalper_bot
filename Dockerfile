FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libpq-dev \
    libgomp1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=Europe/Berlin
CMD ["python", "scalper_bot.py"]