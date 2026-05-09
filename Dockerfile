FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.32.0 \
    httpx==0.27.2

COPY shim.py .

EXPOSE 3100

CMD ["uvicorn", "shim:app", "--host", "0.0.0.0", "--port", "3100", "--no-access-log"]
