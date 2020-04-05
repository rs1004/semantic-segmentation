FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt update && apt -y upgrade && apt-get update && apt-get -y upgrade \
    && pip install --upgrade pip \
        Pillow==7.1.1 \
        tqdm==4.45.0

WORKDIR /work
