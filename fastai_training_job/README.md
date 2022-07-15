## fastai multi-class multi-label classififcation example

## Local training

TBA

## GCP training on vertexai

### 1- Build image
```
IMAGE_NAME='{trainer_image_tagging}'
IMAGE_TAG='vertexai_train_job'
PROJECT_ID='{valid_gcp_project_name}}'
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG
TRAINING_APP_FOLDER=train_job_image

gcloud builds submit --tag $IMAGE_URI $TRAINING_APP_FOLDER
```

### 2- Trigger training job 

To trigger a sample training job use:

```
gcloud beta ai custom-jobs create \
--display-name='trainvertex_job_minimal' \
--region=europe-west4 \
--project='{valid_gcp_project_name}}' \
--config=vertexai_config_test.yml 
```
