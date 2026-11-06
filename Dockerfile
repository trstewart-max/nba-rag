FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY nba_rag /app/nba_rag
COPY eval /app/eval
COPY docs /app/docs
CMD ["uvicorn", "nba_rag.server:app", "--host", "0.0.0.0", "--port", "8000"]
