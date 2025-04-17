# Use Python slim as base image
FROM python:3.11-slim

# Avoid prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OCR and PDF handling
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install gunicorn

# Expose port for Render (or general web service)
EXPOSE 5000

# Run Flask app with gunicorn in production mode
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
