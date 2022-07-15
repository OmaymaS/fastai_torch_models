# fastai vs. native torch predictions

## Overview

fastai is a high level API built on top of torch providing features to simplify training models. Everything written in fastai can be converted to native torch. This could be useful if a decision is made to reducde dependencies in production *(e.g. prediction part)* .

The included scripts and notebook give an example of prediction/inference in fastai and native torch.

## Getting started

To be able to use the following scripts, add the following:

**Models**
- add a model trained using fastai and exported as `.pkl`  under `/models`.
- add the corresponding `.jit` model under `/models`.

Note: you can use the code under [fastai_training_job](https://github.com/OmaymaS/fastai_torch_models/tree/main/fastai_training_job) to train a model.

**Images**
- add few images under `images/{your_dir}`.

## Examples (scripts)

### fastai batch prediction 

- Create a virtual environment and install requirements

```
python -m venv .venv_fastai
source .venv_fastai/bin/activate
pip install -r requirements_fastai.txt
```

- Run `python predict_fastai.py` with your parameters

```
python predict_fastai.py \
 --model_path=models \
 --model_name=model.pkl \
 --test_data_path=images/images_zz \ 
 --batch_size=16
```

### torch batch prediction 

To test the following exampl:

- Create a virtual environment and install requirements (does not include fastai, just native torch).

```
python -m venv .venv_torch
source .venv_torch/bin/activate
pip install -r requirements_torch.txt
```

- Run the following *(same as `./predict_torch.py`)*. Note that the `--image_resize_value` should be specified unlike in fastai case.

```
python predict_fastai.py \
 --model_path=models \
 --model_name=model.pt \
 --test_data_path=images/images_zz \
 --image_resize_value=460
 --batch_size=16
```
## Further explanation and details (notebook)

- Set the virtual environment as the IPython kernel for Jupyter notebook execution

```
ipython kernel install --user --name=.venv_fastai
```

- Launch the notebook and follow the explanation or use your model to run the chunks.
