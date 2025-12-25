# -----------------------------
# Base image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy requirements first
# -----------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy application code ONLY
# -----------------------------
COPY deployment/ deployment/
COPY src/ src/

# -----------------------------
# Expose Streamlit port
# -----------------------------
EXPOSE 8501

# -----------------------------
# Run Streamlit app
# -----------------------------
CMD ["streamlit", "run", "deployment/claud_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
