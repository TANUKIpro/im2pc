# Pi3X + SAM2 UI Container
# This container runs the Gradio web UI and communicates with the host inference server

FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code is mounted via docker-compose volumes
# This allows for development without rebuilding

# Expose Gradio port
EXPOSE 7860

# Environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV INFERENCE_SERVER_URL=http://host.docker.internal:5050

# Run the application
CMD ["python", "main.py"]
