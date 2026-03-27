# HeRoS Docker Image
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src src
COPY configs configs
COPY tests tests

ENV PYTHONPATH=/app
ENV HEROS_VERSION=0.1.0

CMD ["python", "-c", "import heros; print(f'HeRoS v{heros.__version__}')"]
