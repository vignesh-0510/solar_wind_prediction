FROM nvcr.io/nvidia/pytorch:24.01-py3
 
COPY dockerfiles/requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

RUN pip install pyhdf sewar optree scipy

RUN mkdir /app

WORKDIR /app