import fire
from fastai.vision.all import *

from test_model_utils import *

## Usage
## - A an independant script for getting metrics on test data
#     --model_path='models/model_v1.pkl' \
#     --testing_images_path='./temp_test/images_test' \
#     --testing_dataset_path='./temp_test/data_test.csv' \
#     --tag_column="tag" \
#     --tags="tag_01, tag_02, tag03" \
#     --test_thresholds="'0.5'" \
#     --output_path='./temp_test/test_metrics.json'


def test_model_step(
    model_path=None,
    testing_images_path=None,
    testing_dataset_path=None,
    tag_column="tag",
    tags=None,
    test_thresholds="'0.5'",
    output_path=None,
):

    ## tag list
    if not tags is None:
        tag_list = [included_tag.strip() for included_tag in tags.split(",")]
    else:
        tag_list = None

    ## threshold list to use
    thr_list = [float(thr.strip()) for thr in test_thresholds.split(",")]

    # load model
    learn_multi = load_learner(model_path)

    ## predict and calculate metrics
    test_metrics = predict_calculate_classification_metrics(
        model_trained=learn_multi,
        testing_images_path=testing_images_path,
        testing_dataset_path=testing_dataset_path,
        tag_column=tag_column,
        tag_list=tag_list,
        test_threshold_list=thr_list,
    )

    with open(output_path, "w") as f:
        json.dump(test_metrics, f)
    print(f"metrics saved: {output_path}")


if __name__ == "__main__":
    fire.Fire(test_model_step)