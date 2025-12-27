FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY deployment/ deployment/
COPY src/ src/
EXPOSE 8501
CMD ["streamlit", "run", "deployment/claud_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
