from setuptools import find_packages, setup

setup(
    name="ml_example",
    packages=find_packages(),
    version="0.1.0",
    description="Example of ml project",
    author="Your name (or your organization/company/team)",
    entry_points={
        "console_scripts": [
            "ml_example_train = ml_example.train_pipeline:train_pipeline_command"
        ]
    },
    install_requires=[
        "click==7.1.2",
        "python-dotenv>=0.5.1",
        "scikit-learn==0.24.1",
        "dataclasses==0.7",
        "pyyaml==3.11",
        "marshmallow-dataclass==8.3.0",
        "pandas==1.1.5",
        "boto3==1.17.49",
        "dvc>=2.0.0",
        "mlflow==1.15.0",
    ],
    license="MIT",
)
