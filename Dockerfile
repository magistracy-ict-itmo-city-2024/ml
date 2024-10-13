FROM python:3.12.6-slim-bullseye

WORKDIR /app

COPY requirements.txt .

RUN pip install --root-user-action=ignore --no-cache-dir -r requirements.txt

COPY main.py /app/main.py
COPY dto.py /app/dto.py
COPY ml.py /app/ml.py

CMD uvicorn main:app --reload --host 0.0.0.0