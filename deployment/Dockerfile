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


# -----------------------------
# Expose Streamlit port
# -----------------------------
EXPOSE 8501

# -----------------------------
<<<<<<< HEAD
# Run Streamlit app
=======
# Run Streamlit (FINAL APP)
>>>>>>> 45e934533720064af228631957909227b83b29f3
# -----------------------------
CMD ["streamlit", "run", "deployment/claud_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
