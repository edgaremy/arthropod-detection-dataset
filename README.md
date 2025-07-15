# French Arthropod Detection Dataset 🐞

### A detection dataset containing labelled images of **French Terrestrial Arthropods**. 

#### Associated Paper: *To be published*

<img src="https://github.com/edgaremy/arthropod-detection-dataset/blob/main/resources/dataset_thumbnail.png?raw=true" width="400" align="center">

The data is extracted from [iNaturalist](https://www.inaturalist.org), and is designed to cover a **wide variety of arthropod families**. The annotation process is documented in this paper *(to be published)*.


## Installation

You first need to setup a Python environment with the correct depedencies.

#### Clone repository

```bash
git clone https://github.com/edgaremy/arthropod-detection-dataset.git
cd arthropod-detection-dataset
```

#### Option #1: Setup Python venv with Conda

- Make sure you first have Conda installed
- Create a new conda virtual env:
```bash
# You can replace "arthropod" by any name you like
conda create --name arthropod python=3.12.1
```
- You can now activate the venv, and install the requirements with pip:
```bash
pip install -r requirements.txt
```

The environment is now ready ! Note that you will need to activate the venv whenever you want to use it (and deactivate it when you're done):
```bash
# Activate the new venv:
conda activate arthropod

# Deactivate
conda deactivate
```

#### Option #2: Use already existing Python venv

Install requirements with pip:
```bash
pip install -r requirements.txt
```


## Download the dataset

The images on iNaturalist can have various copyrights. As such, they cannot be directly provided, but can be downloaded with our script.

```bash
python dataset_src/download_dataset.py
```

The `dataset_src/download_dataset.py` script downloads images from iNaturalist and create the structure of the yolo dataset accordingly with the labels stored in `resources/dataset_labels.zip`

*Note:* You can also download additional validation data, that was used to assess the generalization capabilities of the detection model.

## Use the detection model

If you want to try the detection model directly, you don't need to download the dataset. ***TODO***

## Going further

#### [Train a model from scratch using the dataset](/training)

#### [Reproduce validation results](/validation)

#
# 🐞 🐜 🦋 🦗 🐝 🕷️ 🐛 🪰 🪲