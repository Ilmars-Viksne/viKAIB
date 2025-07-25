# Micro-SAM on Jetson TX2 with Docker

This project provides a Docker solution to run Micro-SAM, the microscopy-focused version of the Segment Anything model, on an NVIDIA Jetson TX2. The provided Python script `msam_fr.py` will be executed automatically when the container starts.

## Prerequisites

*   An NVIDIA Jetson TX2 with JetPack 4.x flashed.
*   Docker installed on the Jetson TX2.

## Files

*   `Dockerfile`: The recipe to build the Docker image with all necessary dependencies.
*   `msam_fr.py`: The Python script that will be executed inside the container.
*   `README.md`: This file.

## Setup

1.  **Save the files:**
    Make sure you have `Dockerfile` and `msam_fr.py` in the same directory on your Jetson TX2.

2.  **Build the Docker image:**
    Open a terminal on your Jetson TX2, navigate to the directory containing the files, and run the following command:

    ```bash
    docker build -t micro-sam-jetson .
    ```

    This process will take some time as it downloads the base image and installs all the required packages.

## Running the Container

To start the container and run the `msam_fr.py` script, use the following command:

```bash
docker run --rm -it --runtime nvidia micro-sam-jetson
