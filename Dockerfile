FROM python:3.6

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY dist/ml_example-0.1.0.tar.gz /ml_example-0.1.0.tar.gz
RUN pip install /ml_example-0.1.0.tar.gz

COPY configs/ /configs
RUN mkdir -p /models

WORKDIR .

CMD ["ml_example_train", "configs/train_config.yaml"]
