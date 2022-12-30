## fastai multi-class multi-label classififcation example

This directory includes an example of a multi-class multi-label classififcation using fastai.

Note that the data is expected to be in the form provided under `sample_data`.

## Local training

An example of triggering training locally using data under `sample_data`. 

**Note** that a small sample is included to show the expected format but not enough for training a proper model.

```
python train_job_image/train.py \
    --job_dir='sample_data' \
    --model_version='v00' \
    --training_images_path='sample_data/images_train_valid' \
    --training_dataset_path='sample_data/data_train_valid.csv' \
    --model_config_path='sample_data/params_config_example.json' 
```

## GCP training on vertexai

An example of bhuilding a GCR image then triggering training on vertexai. 

### 1- Build image

```
PROJECT_NAME='add_valid_project_name' # replace
IMAGE_NAME='trainer_image_tagging'
IMAGE_TAG='vertexai_train_job_test'
IMAGE_URI=gcr.io/$PROJECT_NAME/$IMAGE_NAME:$IMAGE_TAG
TRAINING_APP_FOLDER=train_job_image 

gcloud builds submit --tag $IMAGE_URI $TRAINING_APP_FOLDER
```

### 2- Trigger training job 

**Expected Data format in GCS**

The training data paremeters  `--training_dataset_path` and `--training_images_path` should refer to `data_train_valid.csv` and raw images prefix `images_train_valid` correspondingly with the format shown under `sample_data`.

```
$DATA_VERSION
│───data_train_valid.csv
│───data_test.csv
│───images_train_valid
│   │   246.jpeg
│   │   304.jpeg
│       ....
└───images_test
    │   2484.jpeg
    │   2104.jpeg
        ....
```

**Triggering the job**

```
DATA_BUCKET= 'data_bucket'
JOB_OUTPUT_PATH=gs://$DATA_BUCKET/temp
DATA_PATH_DEFAULT=gs://$DATA_BUCKET/train_test_data_versions
DATA_VERSION='20221201'
HYPERPARAMS_PATH_DEFAULT='gs://$DATA_BUCKET/params_config_example.json'

gcloud beta ai custom-jobs create \
    --display-name='trainvertex_job_minimal' \
    --region=europe-west4 \
    --project=$PROJECT_NAME \
    --worker-pool-spec=replica-count=1,machine-type='n1-highmem-2',accelerator-type='NVIDIA_TESLA_T4',accelerator-count=1,container-image-uri=${IMAGE_URI} \
    --args='--job_dir'=$JOB_OUTPUT_PATH,'--training_dataset_path'=$DATA_PATH_DEFAULT/$DATA_VERSION/data_train_valid.csv,'--training_images_path'=$DATA_PATH_DEFAULT/$DATA_VERSION/images_train_valid,'--model_hyperparms_path'=$HYPERPARAMS_PATH_DEFAULT,'--model_version'='v_00'
```
