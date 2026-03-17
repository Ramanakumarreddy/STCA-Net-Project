# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Update the package list and install system dependencies
RUN apt-get update -y && \
    apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxinerama1 \
    libxi6 \
    mesa-common-dev \
    libegl1-mesa \
    libegl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables for CPU operation
ENV MEDIAPIPE_DISABLE_GPU=1
ENV KMP_DUPLICATE_LIB_OK=True
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4
ENV VECLIB_MAXIMUM_THREADS=4

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the application will run on (optional)
EXPOSE 10000

# Run the application
CMD ["python", "server.py"]
