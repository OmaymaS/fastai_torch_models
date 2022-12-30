## fastai multi-class multi-label classififcation example

## Local training

```
python train_job_image/train.py \
    --job_dir=./temp_test/ \
    --model_version='v00' \
    --training_images_path='./temp_test/images_train_valid' \
    --training_dataset_path='./temp_test/data_train_valid.csv' \
    --model_hyperparms_path='params_config_example.json' \
    --test_step=True \
    --testing_images_path='./temp_test/images_test' \
    --testing_dataset_path='./temp_test/data_test.csv' 
```

## GCP training on vertexai

### 1- Build image
****
```
PROJECT_NAME='add_valid_project_name'
IMAGE_NAME='trainer_image_tagging'
IMAGE_TAG='vertexai_train_job_test'
IMAGE_URI=gcr.io/$PROJECT_NAME/$IMAGE_NAME:$IMAGE_TAG
TRAINING_APP_FOLDER=train_job_image 

gcloud builds submit --tag $IMAGE_URI $TRAINING_APP_FOLDER
```

### 2- Trigger training job 

To trigger a sample training job use:

```
DATA_BUCKET= 'data_bucket'
JOB_OUTPUT_PATH=gs://$DATA_BUCKET/temp
DATA_PATH_DEFAULT=gs://$DATA_BUCKET/train_test_data_versions
DATA_VERSION='20221201'
HYPERPARAMS_PATH_DEFAULT='gs://$DATA_BUCKET/model_hyperparams/params_config_example.json'

gcloud beta ai custom-jobs create \
    --display-name='trainvertex_job_minimal' \
    --region=europe-west4 \
    --project=$PROJECT_NAME \
    --worker-pool-spec=replica-count=1,machine-type='n1-highmem-2',accelerator-type='NVIDIA_TESLA_T4',accelerator-count=1,container-image-uri=${IMAGE_URI} \
    --args='--job_dir'=$JOB_OUTPUT_PATH,'--training_dataset_path'=$DATA_PATH_DEFAULT/$DATA_VERSION/data_train_valid.csv,'--training_images_path'=$DATA_PATH_DEFAULT/$DATA_VERSION/images_train_valid,'--model_hyperparms_path'=$HYPERPARAMS_PATH_DEFAULT,'--model_version'='v_00','--testing_dataset_path'=$DATA_PATH_DEFAULT/$DATA_VERSION/data_test.csv,'--testing_images_path'=$DATA_PATH_DEFAULT/$DATA_VERSION/images_test,'--test_step'='True' 
```
