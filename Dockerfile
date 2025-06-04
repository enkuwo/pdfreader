FROM python:3.11-slim

# Install system packages
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app files
COPY . .

# Start the app
CMD ["gunicorn", "app:app"]