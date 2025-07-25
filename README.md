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


# CI/CD Deployment with GithubActions & AWS

## 1. Login to your AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is a virtual machine

	2. ECR: Elastic Container Registry to save your docker image in AWS


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR to EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 520061828458.dkr.ecr.ap-south-1.amazonaws.com/kidney_disease_classifier

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
## 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


## 7. Setup github secrets:

    AWS_ACCESS_KEY_ID

    AWS_SECRET_ACCESS_KEY

    AWS_REGION

    AWS_ECR_LOGIN_URI =  566373416292.dkr.ecr.ap-south-1.amazonaws.com (demo)

    ECR_REPOSITORY_NAME = simple-app

    (Fill in your above credentials via GitHub Actions' settings)
