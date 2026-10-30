FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY data/download.sh ./data/download.sh
RUN pip install --no-cache-dir . \
 && bash data/download.sh \
 && python -m smartdischarge.train --data data/diabetic_data.csv --out reports
EXPOSE 8000
CMD ["uvicorn", "smartdischarge.api:app", "--host", "0.0.0.0", "--port", "8000"]
