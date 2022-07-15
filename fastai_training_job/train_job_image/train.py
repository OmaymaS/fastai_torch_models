import json
import os
import time

import fire
import numpy as np
import pandas as pd
from fastai.vision.all import *

from gcp_utils import publish_metrics

timestamp = time.strftime("%Y%m%d-%H%M%S")


def train_evaluate(job_dir: str = None,
                   training_dataset_path: str = None,
                   training_images_path: str = None,
                   model_version: str = None,
                   model_hyperparms_path: str = None):

    if torch.cuda.is_available():
        print('GPU available')

    # GCS gets mounted when the instance starts using gcsfuse. All prefixes exist under /gcs/
    job_dir, training_images_path, model_hyperparms_path = [f.replace(
        'gs://', '/gcs/') if f.startswith('gs') else f for f in [job_dir, training_images_path, model_hyperparms_path]]

    # create subdir/prefix to export results
    job_subdir_export = f'{job_dir}/{model_version}_{timestamp}'
    os.makedirs(job_subdir_export)

    # read hyperparams values
    with open(model_hyperparms_path) as model_hparams_json:
        model_hparams_dict = json.load(model_hparams_json)
        print(f'hyperparameters: {model_hparams_dict}')

    # read training data csv
    print('Reading csv ...')
    df_train = pd.read_csv(training_dataset_path)

    print('Loading and transforming images...')
    # define datablock
    img_block_01 = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                             splitter=ColSplitter('is_valid'),
                             get_x=ColReader(
                                 'image_id', pref=f'{training_images_path}/', suff=''),
                             get_y=ColReader(
                                 model_hparams_dict['TAG_COLUMN'], label_delim=','),
                             item_tfms=Resize(
                                 model_hparams_dict['RESIZE_VALUE'], method=model_hparams_dict['METHOD']),
                             batch_tfms=aug_transforms(do_flip=False))

    # load images and transform
    img_dls_01 = img_block_01.dataloaders(df_train)

    # train
    print('Starting training...')
    fscore = F1ScoreMulti()
    learn_01 = cnn_learner(dls=img_dls_01,
                           arch=eval(model_hparams_dict['ARCH_RESNET']),
                           metrics=[accuracy_multi, fscore])
    learn_01.fine_tune(
        model_hparams_dict['EPOCHS'],
        base_lr=model_hparams_dict['BASE_LR'],
        freeze_epochs=model_hparams_dict['FREEZE_EPOCHS'],
        cbs=[SaveModelCallback(monitor='accuracy_multi',
                               fname='model', with_opt=True)]
    )
    print('Training completed.')

    # dict to add metrics
    metrics_log = {}

    # get final model metrics
    # here accuracy_multi is maximized (can be changed to any other metric like valid_loss)
    accuracy_multi_train = learn_01.recorder.metrics[0].value.item()
    metrics_log['accuracy_multi_train'] = accuracy_multi_train

    # save metrics
    print("Saving metrics")
    with open(f'{job_subdir_export}/metrics_log.json', "w") as f:
        json.dump(metrics_log, f)
    print(metrics_log)

    # pickle model ------
    model_name = f'model_{model_version}'

    pickle_model_path = f'{job_subdir_export}/{model_name}.pkl'
    print(f'Saving pickle model to {pickle_model_path}')
    learn_01.export(f'{pickle_model_path}')

    # jit model ----------
    # save for native torch import later
    dummy_inp = torch.randn(
        [1, 3, model_hparams_dict['RESIZE_VALUE'], model_hparams_dict['RESIZE_VALUE']])  # dummy

    jit_model_path = f'{job_subdir_export}/{model_name}.pt'
    print(f'Saving jit model to {jit_model_path}')
    torch.jit.save(torch.jit.trace(learn_01.model, dummy_inp),
                   f'{job_subdir_export}/model_{model_version}.pt')

    # save vocab with jit model
    vocab = np.save('models/vocab.npy', learn_01.dls.vocab)


if __name__ == "__main__":
    fire.Fire(train_evaluate)
