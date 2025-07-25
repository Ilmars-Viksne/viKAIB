# Base image for Jetson TX2 with L4T from NVIDIA
FROM nvcr.io/nvidia/l4t-base:r32.7.1

# Set the working directory inside the container
WORKDIR /app

# Install essential system-level packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install a compatible version of PyTorch for Jetson TX2 (JetPack 4.x)
# This specific wheel is for Python 3.6, the default on this JetPack version
RUN wget https://nvidia.box.com/shared/static/mmu3xb3sp4o8qg9tji90kkxl1eijjfc6.whl -O torch-1.1.0-cp36-cp36m-linux_aarch64.whl && \
    pip3 install torch-1.1.0-cp36-cp36m-linux_aarch64.whl && \
    rm torch-1.1.0-cp36-cp36m-linux_aarch64.whl

# Install other required Python libraries
RUN pip3 install \
    numpy \
    matplotlib \
    scikit-image \
    tqdm \
    opencv-python

# Install Micro-SAM directly from its GitHub repository
RUN pip3 install git+https://github.com/computational-cell-analytics/micro-sam.git

# Copy the provided Python script into the container's working directory
COPY msam_fr.py .

# Set the entrypoint to execute the Python script when the container starts
ENTRYPOINT ["python3", "msam_fr.py"]