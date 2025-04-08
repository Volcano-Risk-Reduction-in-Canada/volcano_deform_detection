FROM ubuntu:22.04

# Install system dependencies for GDAL & OpenGL
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    libgl1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_VERSION=3.10.2

ENV CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES


COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Set the working directory to /app
WORKDIR /app

# Copy the files in the current directory (your local project files) to /app inside the container
COPY compare_disp_wrp_images.py . 

COPY calculate_ai_confidence.py . 
COPY data_utils.py . 
COPY get_probability_map_func_single_resolution.py .
COPY get_disp_and_wrp_images_from_s3.py .
# copy the AI models
COPY models /app/models

# Specify the command to run your script
# This will be executed when the container starts
ENTRYPOINT ["python3", "compare_disp_wrp_images.py"]