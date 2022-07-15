import argparse

from fastai.vision.all import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='models')
parser.add_argument('--model_name', default='model_modified.pkl')
parser.add_argument('--test_data_path', default='images/imgs_zz/')
parser.add_argument('--batch_size', default=64)
args = parser.parse_args()

PRED_NUM_WORKERS = 0  # for local tests
MODEL_PATH = args.model_path
MODEL_NAME = args.model_name
TEST_DATA_PATH = args.test_data_path
BATCH_SIZE = args.batch_size

# load model
learn_multi = load_learner(f'{MODEL_PATH}/{MODEL_NAME}')

# predict on batches
tst_files = get_image_files(TEST_DATA_PATH)
tst_dl = learn_multi.dls.test_dl(
    tst_files, bs=BATCH_SIZE, num_workers=PRED_NUM_WORKERS)
preds_fastai, _ = learn_multi.get_preds(dl=tst_dl)

print(preds_fastai)
