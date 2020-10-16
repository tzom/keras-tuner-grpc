ARG PY_VERSION=3.7
FROM python:${PY_VERSION}

RUN pip install tensorflow-cpu \
                tensorflow_datasets
                future \
                numpy \
                requests \
                keras-tuner    

RUN apt-get update
RUN apt-get install -y git
RUN git clone https://github.com/keras-team/keras-tuner-grpc.git

ENV KERASTUNER_TUNER_ID "chief"
ENV KERASTUNER_ORACLE_IP "127.0.0.1"
ENV KERASTUNER_ORACLE_PORT "8000"
EXPOSE 8000

