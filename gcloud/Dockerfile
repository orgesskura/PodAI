FROM python:3.9-slim

# Install Node.js and npm
RUN apt-get update && apt-get install -y nodejs npm git

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install frontend dependencies and build React app
WORKDIR /app/lex-chat
RUN npm install
RUN npm install react-router-dom
RUN npm run build

# Move back to the main directory
WORKDIR /app

# Define environment variable
ENV PORT 8080

# Run app.py when the container launches
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app