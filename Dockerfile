# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy trainer.py
COPY trainer.py ./trainer.py

# Copy requirements.txt
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point
ENTRYPOINT ["python", "trainer.py"]
