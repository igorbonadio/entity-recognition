FROM ubuntu:18.04

LABEL maintainer "Igor Bonadio <igorbonadio@gmail.com>"

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get update \
    && apt-get install -y python3-dev python3-pip git \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && ln -sf /usr/bin/python3 /usr/bin/python

ADD dist_model /entity-recognition/dist_model
ADD api.py /entity-recognition/api.py
ADD requirements.txt /entity-recognition/requirements.txt

WORKDIR /entity-recognition

RUN pip install -r requirements.txt

EXPOSE 8080

CMD gunicorn api --bind localhost:8080 --worker-class aiohttp.GunicornWebWorker --workers=4 --preload