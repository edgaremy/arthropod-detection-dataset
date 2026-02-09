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
pip install ultralytics wget huggingface_hub seaborn pingouin pypalettes
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
conda activate arthropod
```
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

The `src/download_dataset.py` script downloads images from iNaturalist and creates the structure of the yolo dataset accordingly with the labels stored in `resources/dataset_labels.zip`. Please note that due to API constraints, the download can be blocked after a while on iNaturalist's server. It that's the case, you should still be able to run the script again later, and it should resume where it stopped. 

*Note:* You can also [download additional validation data](/validation/README.md#download-additional-validation-datasets), that was used to assess the generalization capabilities of the detection model.

<br />

## Use the detection model

If you just want to try the detection model directly, you don't need to download the dataset. The models are automatically downloaded from Hugging Face Hub.

### Quick start - Command line

```bash
# Process a single image (uses YOLO11n PyTorch model by default)
python inference/run_model_with_cmd.py path/to/image.jpg

# Use YOLO11l model with ONNX format and save all outputs
python inference/run_model_with_cmd.py images/ \
    --format onnx --model-size l \
    --save-crops --save-labels --save-bbox-view

# Use GPU and custom confidence threshold
python inference/run_model_with_cmd.py images/ --device cuda --conf 0.5
```

For complete documentation, see **[inference/README.md](inference/README.md)**

### Use as Python module

```python
from inference.run_model_with_cmd import load_model_from_huggingface, run_inference

# Load model (automatically downloads from Hugging Face Hub)
model = load_model_from_huggingface(
    repo_id="edgaremy/arthropod-detector",
    filename="yolo11l_ArthroNat+flatbug.onnx"  # or .pt for PyTorch
)

# Run inference with all output options
summary = run_inference(
    model=model,
    input_path="images/",
    results_folder="results",
    save_crops=True,
    save_labels=True,
    save_bbox_view=True,
    conf_threshold=0.5,
    device="cuda"
)
```

**Available models:**
- `yolo11n_ArthroNat+flatbug.pt` / `.onnx` - Nano (fastest)
- `yolo11l_ArthroNat+flatbug.pt` / `.onnx` - Large (most accurate)

For more details, check out:
- **[inference/README.md](inference/README.md)** - Comprehensive usage guide
- **[inference/example_module_usage.py](inference/example_module_usage.py)** - Python module examples
- The dedicated **Hugging Face [Model Repo](https://huggingface.co/edgaremy/arthropod-detector)** 🤗
- [Ultralytics Documentation](https://docs.ultralytics.com/)

<br />

## Going further

#### [Train a model from scratch using the dataset](/training)

#### [Reproduce validation results](/validation)

#
# [🐞](https://www.gbif.org/species/165599324)  [🐜](https://www.gbif.org/species/4342)  [🦋](https://www.gbif.org/species/797)  [🦗](https://www.gbif.org/species/1718308)  [🐝](https://www.gbif.org/species/1341976)  [🕷️](https://www.gbif.org/species/1496)  [🐛](https://www.gbif.org/species/797)  [🪰](https://www.gbif.org/species/1524843)  [🪲](https://www.gbif.org/species/1043502)
