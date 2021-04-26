FROM python:3.6

RUN mkdir -p /build
COPY dist/ml_example-0.1.0.tar.gz /build/ml_example-0.1.0.tar.gz
RUN pip install /build/ml_example-0.1.0.tar.gz

COPY configs/ /configs
RUN mkdir -p /models

WORKDIR .

ENTRYPOINT ["ml_example_train", "configs/train_config.yaml"]
