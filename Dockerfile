# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user
RUN useradd --create-home --shell /bin/bash fraudguard
RUN chown -R fraudguard:fraudguard /app
USER fraudguard

# Expose ports for both backend and frontend
EXPOSE 8000 8501

# Create startup script
COPY --chown=fraudguard:fraudguard <<EOF /app/start.sh
#!/bin/bash
set -e

echo "Starting FraudGuard MVP..."

# Start the backend API in the background
echo "Starting FastAPI backend on port 8000..."
cd /app/backend
python main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 5

# Start the Streamlit frontend
echo "Starting Streamlit frontend on port 8501..."
cd /app/frontend
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
EOF

RUN chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]
