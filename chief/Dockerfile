ARG PY_VERSION=3.7
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

RUN export KERASTUNER_ORACLE_IP="$(hostname -i)"

ENV KERASTUNER_TUNER_ID "chief"
#ENV KERASTUNER_ORACLE_IP ${IP}
ENV KERASTUNER_ORACLE_PORT "8000"
EXPOSE 8000

