import argparse
import glob

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset_custom import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='models')
parser.add_argument('--model_name', default='model_modified_jit.pt')
parser.add_argument('--test_data_path', default='images/imgs_zz/')
parser.add_argument('--image_resize_value', default=460)
parser.add_argument('--batch_size', default=64)
args = parser.parse_args()

PRED_NUM_WORKERS = 0  # for local tests
MODEL_PATH = args.model_path
MODEL_NAME = args.model_name
TEST_DATA_PATH = args.test_data_path
IMG_RESIZE_VALUE = args.image_resize_value
BATCH_SIZE = args.batch_size

# load model
model_loaded = torch.jit.load(f'{MODEL_PATH}/{MODEL_NAME}')
# as per the docs, eval() is required after load() during inference/prediction
model_loaded.eval()


test_files = glob.glob(f'{TEST_DATA_PATH}*')

# predict on batches
dataset = ImageDataset(test_files, img_size=IMG_RESIZE_VALUE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                    num_workers=PRED_NUM_WORKERS)

final_output = []
with torch.no_grad():
    for batch in loader:
        preds = model_loaded(batch)
        preds_batch = preds.sigmoid()  # multi-label multi-class
        final_output.append(preds_batch)
preds_torch = torch.cat(final_output, dim=0)

print(preds_torch)
