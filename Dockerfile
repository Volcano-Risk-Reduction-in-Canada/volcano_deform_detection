FROM python:3.9-slim

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt
RUN apt-get update && apt-get install libgl1
RUN apt-get install libgdal-dev

# Copy the current directory (your local project files) to /app inside the container
COPY . /app

# Step 5: Set the working directory to /app
WORKDIR /app

# Step 6: Specify the command to run your script
# This will be executed when the container starts
ENTRYPOINT ["python3", "get_probability_diff_resolutions.py"]