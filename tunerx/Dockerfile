ARG PY_VERSION=3.7
ARG TUNERX=0
FROM python:${PY_VERSION}

RUN pip install tensorflow-cpu \
                tensorflow_datasets \
                future \
                numpy \
                requests \
                keras-tuner    

RUN apt-get update
RUN apt-get install -y git
RUN git clone https://github.com/tzom/keras-tuner-grpc.git

ENV KERASTUNER_TUNER_ID "tuner"${TUNERX}
ENV KERASTUNER_ORACLE_IP "0.0.0.0"
ENV KERASTUNER_ORACLE_PORT "8000"
