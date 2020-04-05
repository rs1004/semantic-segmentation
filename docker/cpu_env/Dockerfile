FROM ubuntu:18.04

RUN apt update && apt -y upgrade && apt-get update && apt-get -y upgrade \
    && apt install -y wget unzip \
    && apt install -y python3 \
    && apt-get install -y python3-pip \
    && pip3 install \
        Pillow==7.1.1 \
        tqdm==4.45.0 \
        numpy==1.18.2 \
    && echo "alias python='python3'" >> ~/.bashrc

WORKDIR /work