FROM python:3.9-slim

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt
RUN apt-get update && apt-get install libgl1
RUN apt-get install libgdal-dev


ENTRYPOINT [ "python3" ]