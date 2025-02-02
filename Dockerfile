FROM ubuntu:latest

# Install required system packages
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# Set the working directory
WORKDIR /app

# Create a virtual environment
RUN python3 -m venv /app/venv

# Activate the virtual environment and install dependencies
COPY requirements.txt .
COPY .env .
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the application file
COPY rag_helper_functions.py .
COPY pdf_upload_rag.py .

# Use the virtual environment's Python to run the app
CMD ["/app/venv/bin/streamlit", "run", "pdf_upload_rag.py", "--server.port=8501", "--server.address=0.0.0.0"]
