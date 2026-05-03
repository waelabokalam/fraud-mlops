FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

RUN mkdir -p models && python3 src/train.py

EXPOSE 8080

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8080"]