# Use the official Python image
FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Run the Streamlit app
CMD ["streamlit", "run", "cat-dog.py", "--server.port=8501", "--server.address=0.0.0.0"]
