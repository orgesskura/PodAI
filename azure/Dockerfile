# Use a Python base image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    sox \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install PyTorch and related packages
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Copy the requirements file into the container
COPY requirements.txt .

# Install Cython first
RUN pip install --no-cache-dir Cython

# Install Python dependencies with looser constraints
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages separately
RUN pip install --no-cache-dir \
    git+https://github.com/m-bain/whisperX.git@78dcfaab51005aa703ee21375f81ed31bc248560 \
    nemo_toolkit[asr]==1.23.0 \
    git+https://github.com/facebookresearch/demucs#egg=demucs \
    git+https://github.com/oliverguhr/deepmultilingualpunctuation.git \
    git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git \
    whisper \
    google-auth-oauthlib \
    google-auth-httplib2 \
    google-api-python-client \
    azure-storage-blob \
    PyDrive \
    nltk

# Download NLTK data
RUN python -m nltk.downloader punkt

# Copy the Python script into the container
COPY diarization_script.py .

# Set the entrypoint command
CMD ["python", "diarization_script.py"]