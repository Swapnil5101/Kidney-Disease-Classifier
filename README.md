# Kidney-Disease-Classifier

## Workflows

1. Update config.yaml
2. Update params.yaml
3. Update the entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline 
7. Update the main.py
8. Update the dvc.yaml
9. Update app.py

## How to Run?
### STEPS:

Clone the repository

```bash
https://github.com/Swapnil5101/Kidney-Disease-Classifier
```
### STEP 01- Create a virtual environment

```bash
# Replace 'venv_name' with name of virtual env you want
python -m venv /venv_name
```

```bash
source ./venv_name/bin/activate
```


### STEP 02- Install the requirements
```bash
pip install -r requirements.txt
```

## MLflow and DagsHub

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

- [DagsHub Documentation](https://dagshub.com/)



MLFLOW_TRACKING_URI=https://dagshub.com/Swapnil5101/Kidney-Disease-Classifier.mlflow

MLFLOW_TRACKING_USERNAME=f4b146a8163da584e740a5b5b884330b40ade30b


Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/Swapnil5101/Kidney-Disease-Classifier.mlflow

export MLFLOW_TRACKING_USERNAME=f4b146a8163da584e740a5b5b884330b40ade30b

```

### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag