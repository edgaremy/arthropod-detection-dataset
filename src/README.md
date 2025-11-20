# Source Scripts

Scripts for dataset preparation and basic inference.

## Dataset Preparation

**`download_dataset.py`** - Downloads arthropod images from iNaturalist using dataset labels and creates YOLO dataset structure.

**`preview_bbox.py`** - Visualizes bounding box annotations on training images for quality control.

**`convert_flatbug.py`** - Converts flatbug segmentation dataset to YOLO detection format with train/val/test splits.

**`split_flatbug.py`** - Splits flatbug validation set into separate validation and test sets (90/10 ratio).

## Configuration Files

**`dataset_images.csv`** - List of image URLs and metadata for dataset download.

**`dataset_labels.zip`** - YOLO format annotation labels for all dataset images.

**`arthro_dataset_hierarchy.csv`** - Taxonomic hierarchy (class, order, family, genus, species and iNaturalist taxon_id) for all species in dataset.

**`dataset_arthro.yaml`**, **`dataset_flatbug.yaml`**, **`datasets.yaml`** - YOLO dataset configuration files.

---

For comprehensive inference capabilities, see [`inference/`](../inference/README.md).
