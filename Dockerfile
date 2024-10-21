# Use an official Python base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy dependency and Python files first
COPY requirements.txt /app/requirements.txt
COPY train.py /app/train.py

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Use ENTRYPOINT to allow the script to receive arguments
ENTRYPOINT ["python", "train.py"]
