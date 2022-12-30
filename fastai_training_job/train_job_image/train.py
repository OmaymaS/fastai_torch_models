import json
import os
import time

import fire
import pandas as pd
from fastai.vision.all import *

from test_model_utils import *

training_timestamp = time.strftime("%Y%m%d-%H%M%S")

TEST_THR_LIST = [x / 10 for x in range(11)]


def train_evaluate(
    job_dir: str = None,
    training_dataset_path: str = None,
    training_images_path: str = None,
    model_version: str = None,
    model_config_path: str = None,
    test_step: bool = False,
    testing_dataset_path: str = None,
    testing_images_path: str = None,
):

    if torch.cuda.is_available():
        print("GPU available")

    ## GCS gets mounted when the instance starts using gcsfuse. All prefixes exist under /gcs/
    job_dir, training_images_path, model_config_path = [
        f.replace("gs://", "/gcs/") if f.startswith("gs://") else f
        for f in [job_dir, training_images_path, model_config_path]
    ]

    ## create subdir/prefix to export results
    job_subdir_gcs_export = f"{job_dir}/{model_version}_{training_timestamp}"
    os.makedirs(job_subdir_gcs_export)

    ## read config values
    with open(model_config_path) as model_config_json:
        model_config_dict = json.load(model_config_json)
        print(f"hyperparameters: {model_config_dict}")

    ## include specified tags if available, otherwise sue all
    if "TAGS" in model_config_dict:
        TAG_LIST = [
            included_tag.strip()
            for included_tag in model_config_dict["TAGS"].split(",")
        ]
    else:
        TAG_LIST = None

    ## read training data csv
    print("Reading csv ...")
    df_train = pd.read_csv(training_dataset_path)
    if TAG_LIST:
        df_train = df_train[df_train["tag"].isin(TAG_LIST)]

    print(f"number of rows in train data: {len(df_train)}")

    print("Loading and transforming images...")
    ## define datablock
    img_block = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock),
        splitter=ColSplitter("is_valid"),
        get_x=ColReader("image_id", pref=f"{training_images_path}/", suff=""),
        get_y=ColReader(model_config_dict["TAG_COLUMN"], label_delim=","),
        item_tfms=Resize(
            model_config_dict["RESIZE_VALUE"], method=model_config_dict["METHOD"]
        ),
        batch_tfms=aug_transforms(do_flip=False),
    )

    ## load images and transform
    img_dls = img_block.dataloaders(df_train)

    ## train
    print("Starting training...")
    fscore = F1ScoreMulti()
    model_multi = cnn_learner(
        dls=img_dls,
        arch=eval(model_config_dict["ARCH_RESNET"]),
        metrics=[accuracy_multi, fscore],
    )
    model_multi.fine_tune(
        model_config_dict["EPOCHS"],
        base_lr=model_config_dict["BASE_LR"],
        freeze_epochs=model_config_dict["FREEZE_EPOCHS"],
        cbs=[SaveModelCallback(monitor="accuracy_multi", fname="model", with_opt=True)],
    )
    print("Training completed.")

    ## dict to add metrics
    metrics_log = {}

    ## get final model metrics
    accuracy_multi_train = model_multi.recorder.metrics[0].value.item()

    ## TEST DATA METRICS -----------------------------------
    ## if test data is provided and test_step is True
    if test_step:
        print("Evaluating model on test data")
        if testing_images_path.startswith("gs://"):
            testing_images_path = testing_images_path.replace("gs://", "/gcs/")

        metrics_test = predict_calculate_classification_metrics(
            model_trained=model_multi,
            testing_images_path=testing_images_path,
            testing_dataset_path=testing_dataset_path,
            tag_column=model_config_dict["TAG_COLUMN"],
            tag_list=TAG_LIST,
            test_threshold_list=TEST_THR_LIST,
        )

        ## append metrics to log
        metrics_log["test_metrics"] = metrics_test
        metrics_log["test_dataset"] = testing_dataset_path
        metrics_log["train_timestamp"] = training_timestamp
        metrics_log["train_metrics"] = {
            "fasti_metrics": {"accuracy_multi_train_default": accuracy_multi_train}
        }
    else:
        metrics_log = {
            "fasti_metrics": {"accuracy_multi_train_default": accuracy_multi_train}
        }
    ## -------------------------------------------------------------

    ## save metrics
    print("Saving metrics")
    metrics_log_path = f"{job_subdir_gcs_export}/metrics_log.json"
    with open(metrics_log_path, "w") as f:
        json.dump(metrics_log, f)

    ## save config
    with open(f"{job_subdir_gcs_export}/model_config.json", "w") as f:
        json.dump(model_config_dict, f)

    ## save model
    model_multi.version = model_version  ## add model version to saved model
    model_name = f"model_{model_version}.pkl"
    model_path = f"{job_subdir_gcs_export}/{model_name}"
    print(f"Saving model to {model_path}")
    model_multi.export(f"{model_path}")

    ## save job details
    job_inputs_log = {
        "job_dir": job_dir,
        "training_dataset_path": training_dataset_path,
        "training_images_path": training_images_path,
        "model_config_path": training_images_path,
        "test_step": test_step,
        "testing_dataset_path": testing_dataset_path,
        "testing_images_path": testing_dataset_path,
    }

    job_inputs_json_path = f"{job_subdir_gcs_export}/job_inputs_log.json"
    with open(job_inputs_json_path, "w") as f:
        json.dump(job_inputs_log, f)

if __name__ == "__main__":
    fire.Fire(train_evaluate)