# French Arthropod Detection Dataset 🐞

### A detection dataset containing labelled images of **French Terrestrial Arthropods**. 

**Associated Paper: *To be published*** \
**Model Page: [huggingface.co/edgaremy/arthropod-detector](https://huggingface.co/edgaremy/arthropod-detector)** 

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/resources/dataset_thumbnail.png?raw=true" width="400" align="center">

The data is extracted from [iNaturalist](https://www.inaturalist.org), and is designed to cover a **wide variety of arthropod families**. The data collection and annotation process is documented in this paper *(to be published)*.

<br />

## Set up Python

### Clone repository

```bash
git clone https://github.com/edgaremy/arthropod-detection-dataset.git
cd arthropod-detection-dataset
```
### Option #1: Quick setup
If you already know how to use Python environments, here are the librairies you need to install:

```bash
pip install ultralytics wget huggingface_hub seaborn pingouin
```
This should suffice, though if you encounter any dependency issues, or want to reproduce the exact setup that was use to get our results, please favor option #2 below.

### Option #2: For reproducibility - Set up Python venv using Conda

- Make sure you first have [conda installed](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Create a new conda virtual env:
```bash
# You can replace "arthropod" by any name you like
conda create --name arthropod python=3.12.11
```
- You can now activate the venv, and install the requirements with pip (*if you already have another non-conda environment, you can do this directly*):
```bash
pip install -r requirements.txt
```

**The environment is now ready !**

*Note:* you will need to activate the venv whenever you want to use it (see [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details).


<br />

## Download the dataset

The images on iNaturalist can have various copyrights. As such, they cannot be directly provided, but can be downloaded using our script.

```bash
python src/download_dataset.py
```

The `src/download_dataset.py` script downloads images from iNaturalist and creates the structure of the yolo dataset accordingly with the labels stored in `resources/dataset_labels.zip`

*Note:* You can also [download additional validation data](/validation/README.md#download-additional-validation-datasets), that was used to assess the generalization capabilities of the detection model.

<br />

## Use the detection model

If you just want to try the detection model directly, you don't need to download the dataset. ***TODO***

### Usage example

Try it using the `src/inference.py` example:
```bash
python src/inference.py <path_to_your_image> --size l --verbose
```
For more details, you can type `python src/inference.py --help`

### Load the model in your Python code

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Download weights from Hugging Face Hub
weights = hf_hub_download(
    repo_id="edgaremy/arthropod-detector",
    filename="yolo11l_arthropod_0.413.pt"
    )

# Load the model with Ultralytics YOLO
model = YOLO(weights)
```

For more details, check out:
- The source code of [`src/inference.py`](src/inference.py)
- The dedicated **Hugging Face [Model Repo](https://huggingface.co/edgaremy/arthropod-detector)** 🤗
- [Ultralytics Documentation](https://docs.ultralytics.com/)

<br />

## Going further

#### [Train a model from scratch using the dataset](/training)

#### [Reproduce validation results](/validation)

#
# 🐞 🐜 🦋 🦗 🐝 🕷️ 🐛 🪰 🪲