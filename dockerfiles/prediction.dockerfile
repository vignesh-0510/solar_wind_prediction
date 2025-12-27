FROM nvcr.io/nvidia/pytorch:24.01-py3
 
COPY dockerfiles/requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

RUN pip install pyhdf sewar optree scipy deepxde accelerate

RUN pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN mkdir /app

WORKDIR /app