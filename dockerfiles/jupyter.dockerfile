FROM nvcr.io/nvidia/pytorch:24.01-py3
 
COPY dockerfiles/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir /app

WORKDIR /app