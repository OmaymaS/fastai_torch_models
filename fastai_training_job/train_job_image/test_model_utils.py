import logging
import os

import pandas as pd
from fastai.vision.all import *
from sklearn.metrics import classification_report


def predict_calculate_classification_metrics(
    model_trained=None,
    testing_images_path=None,
    testing_dataset_path=None,
    tag_column="tag",
    tag_list=None,
    test_threshold_list=[0.5],
):
    """
    High level function to combine prediction and reporting metrics.

    :param model_trained: loaded model (fastai), defaults to None
    :param testing_images_path: directory including at least one image, defaults to None
    :param testing_dataset_path: path to `csv` including at least two columns (`image_id`, `tag`). The value accepts a gcs path as `gs://` when authenticated, defaults to None
    :param tag_column: column name including ground truth labels, defaults to "tag"
    :param tag_list: list of tags to include, defaults to None
    :param test_threshold_list: list of threhsold values to use for calculating classification metrics, defaults to [0.5]
    :return: dict with metrics corresponding to thresholds in `test_threshold_list`
    :rtype: dict
    """

    ## predict probabilities
    test_pred_proba_wide = predict_tag_proba(
        model_trained=model_trained, testing_images_path=testing_images_path
    )

    ## reshape true labels data
    true_labels_long = pd.read_csv(testing_dataset_path)
    true_labels_wide = format_tag_data_wide(
        dataset_df=true_labels_long, tag_column=tag_column
    )
    ## calculate metrics
    if tag_list is None:
        tag_list = model_trained.dls.vocab

    metrics_test = calculate_classififcation_metrics(
        df_true=true_labels_wide,
        df_pred=test_pred_proba_wide,
        tag_list=tag_list,
        test_threshold_list=test_threshold_list,
    )

    return metrics_test


def predict_tag_proba(model_trained=None, testing_images_path: str = None):
    """
    Return predicted probabilities of a set of images under a given directory. In case of a large dataset, the execution will take significantly more time unless executed on an instance with GPU.

    :param model_trained: loaded model (fastai), defaults to None.
    :param testing_images_path: directory including at least one image, defaults to None.
    :return: a dataframe including one row per image and one column per tag.
    :rtype: pd.DataFrame

    :Example:

    >>> from fastai.vision.all import *
    >>> learn_multi = load_learner('model_v_4.0.pkl') ## load sample model `model_v_4.0.pkl` trained and exported using fastai
    >>> predict_tag_proba(learn_multi, testing_images_path='imgs_zz') ## `imgs_zz` a directory with at least one image.

    image_id | tag_01 | tag_02 | tag_0n
    ---------|--------|--------|-------
    1006.jpeg| 0.91   | 0.865  | 0.67
    2349.jpeg| 0.81   | 0.432  | 0.821
    ...      | ...    | ...    | ...
    """

    if len(os.listdir(testing_images_path)) == 0:
        print(
            "testing_images_path does not include images. Please pass a directory including at least one image."
        )
        return

    tst_files = get_image_files(testing_images_path)  ## load images
    tst_dl = model_trained.dls.test_dl(tst_files)
    preds, _ = model_trained.get_preds(dl=tst_dl)  ## predict
    pred_tf_bin = preds.float().numpy()  ## transform data
    pred_df = pd.DataFrame(pred_tf_bin, columns=model_trained.dls.vocab)
    pred_df["image_id"] = [
        str(test_image).split("/")[-1] for test_image in tst_dl.dataset.items
    ]  # add image id in the right order
    return pred_df


def format_tag_data_wide(dataset_df: pd.DataFrame = None, tag_column: str = "tag"):
    """
    Transform ground truth data to wide format (one hot encoded).

    :param dataset_df: dataframe including at least two columns (`image_id`, `tag`), defaults to None
    :param tag_column: column name including ground truth labels, defaults to 'tag'
    :return: dataframe including one row per image and one column per tag.
    :rtype: pd.DataFrame

    :Example:

    >>> data_dict = {'image_id': ['1001.jpeg', '3836.jpeg'],
                     'tag': ['tag_01,tag_02', 'tag_2']}
    >>> df_long = pd.DataFrame(data=data_dict)
    >>> format_tag_data_wide(dataset_df=df_long, tag_column='tag')

    image_id | ... | tag_01      | tag_02
    ---------|-----|-------------|------------
    1001.jpeg| ... | 1           | 1
    3836.jpeg| ... | 0           | 1
    """
    dataset_df[tag_column] = dataset_df[tag_column].apply(lambda x: x.replace(" ", ""))
    tags_wide = dataset_df[tag_column].str.get_dummies(",")
    df_test_one_hot_encoded = pd.concat([dataset_df, tags_wide], axis=1)
    return df_test_one_hot_encoded


def calculate_classififcation_metrics(
    df_true: pd.DataFrame = None,
    df_pred: pd.DataFrame = None,
    tag_list: list = None,
    test_threshold_list: list = [0.5],
):
    """
    Calculate classififcation metrics at one or multiple thresholds

    :param df_true: dataframe with ground truth values in a 1-hot encoded format (one column per tag), defaults to None.
    :param df_pred: dataframe with predicted values (one column per tag), defaults to None.
    :param tag_list: list of tags to include. Any other columns with additional tags will be discarded, defaults to None.
    :param test_threshold_list: list of threhsold values to use for calculating classification metrics, defaults to [0.5].
    :return: dict with metrics corresponding to thresholds in `test_threshold_list`.
    :rtype: dict

    :Example:

    >>> true_labels = pd.DataFrame({'image_id': ['1001.jpeg', '3836.jpeg', '3838.jpeg', '24746.jpeg', '5363.jpeg'],
                                    'tag_01': [1, 0, 0, 0, 1],
                                    'tag_02': [0, 1, 0, 0, 0],
                                    'tag_03': [0, 0 ,1, 0, 0]})

    >>> pred_proba = pd.DataFrame({'image_id':  ['1001.jpeg', '3836.jpeg', '3838.jpeg', '24746.jpeg', '5363.jpeg'],
                                    'tag_01': [0.91, 0.02, 0.04, 0.15, 0.80],
                                    'tag_02':  [0.10, 0.99, 0.01, 0.01, 0.00],
                                    'tag_03':  [0.91, 0.00, 0.96, 0.01, 0.1]})

    >>> calculate_classififcation_metrics(df_true=true_labels,
                                          df_pred=pred_proba,
                                          tag_list=['tag_01', 'tag_02', 'tag_03'],
                                          test_threshold_list=[0.9])
    """
    df_true_pred = df_true.merge(
        df_pred, how="left", on="image_id", suffixes=("_x", "_y")
    )
    y_true = df_true_pred[[tag + "_x" for tag in tag_list]].to_numpy()
    y_pred_proba = df_true_pred[[tag + "_y" for tag in tag_list]].to_numpy()

    metrics_calculated_dict = {}
    for test_thr in test_threshold_list:
        y_pred = 1 * (y_pred_proba >= test_thr)
        # use accuracy_multi() from fastai(required tensors)
        fastai_accuracy_multi = accuracy_multi(
            torch.tensor(y_pred).float(), torch.tensor(y_true).float(), sigmoid=False
        ).tolist()

        sklearn_metrics = classification_report(
            y_true, y_pred, target_names=tag_list, output_dict=True, zero_division=0
        )

        metrics_calculated = {
            "sklearn_metrics": sklearn_metrics,
            "fastai_metrics": {"accuracy_multi_test": fastai_accuracy_multi},
        }
        metrics_calculated_dict[test_thr] = metrics_calculated

    return metrics_calculated_dict