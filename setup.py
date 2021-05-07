from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


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
    install_requires=required,
    license="MIT",
)
