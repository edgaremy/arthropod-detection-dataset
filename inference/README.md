# YOLO Model Inference from Hugging Face - Usage Guide

## 📋 Overview

`run_model_with_cmd.py` automatically downloads and runs arthropod detection models from Hugging Face Hub (`edgaremy/arthropod-detector`) that were trained on this repo dataset. It supports both PyTorch (.pt) and ONNX formats with flexible output options.

**Files in this directory:**
- `run_model_with_cmd.py` - Main inference script (CLI + Python module)
- `example_module_usage.py` - Comprehensive examples for programmatic usage
- `README.md` - This documentation


## 📦 Requirements

If you don't already have a environment set up for the rest of this repo, you can simply add these packages to your python environment.

```bash
pip install ultralytics huggingface-hub
```



## 🚀 Command Line Usage

### **Basic Examples**

```bash
# 1. Simplest usage - single image with default settings (YOLO11n, PyTorch)
python run_model_with_cmd.py image.jpg

# 2. Process folder of images
python run_model_with_cmd.py images/

# 3. Use YOLO11l (large) model instead of nano
python run_model_with_cmd.py images/ --model-size l

# 4. Use ONNX format instead of PyTorch
python run_model_with_cmd.py images/ --format onnx

# 5. Combine ONNX with large model
python run_model_with_cmd.py images/ --format onnx --model-size l
```

### **With Output Options**

```bash
# Save all outputs: crops, labels, and annotated images
python run_model_with_cmd.py images/ \
    --save-crops \
    --save-labels \
    --save-bbox-view

# Save only cropped detections and labels
python run_model_with_cmd.py images/ \
    --save-crops \
    --save-labels

# Save only annotated images with bounding boxes
python run_model_with_cmd.py images/ \
    --save-bbox-view
```

### **Custom Configuration**

```bash
# Custom confidence threshold (default: 0.25)
python run_model_with_cmd.py images/ --conf 0.5

# Use GPU for inference
python run_model_with_cmd.py images/ --device cuda

# Specify GPU device
python run_model_with_cmd.py images/ --device 0

# Custom output folder
python run_model_with_cmd.py images/ --results-folder output/experiment1

# Enable verbose logging
python run_model_with_cmd.py images/ -v
```

### **Complete Example**

```bash
# Full-featured inference run
python run_model_with_cmd.py dataset/images/test/ \
    --format onnx \
    --model-size l \
    --save-crops \
    --save-labels \
    --save-bbox-view \
    --conf 0.5 \
    --device cuda \
    --results-folder results/yolo11l_onnx_conf05 \
    --verbose
```


## 🐍 Python Module Usage

### **Quick Start - See `example_module_usage.py`**

The repository includes a comprehensive example file with 5 different usage patterns:

```bash
# View all examples
python example_module_usage.py

# Edit the file to uncomment and run specific examples
```

### **Basic Usage**

```python
from run_model_with_cmd import load_model_from_huggingface, run_inference

# Load YOLO11n PyTorch model
model = load_model_from_huggingface(
    repo_id="edgaremy/arthropod-detector",
    filename="yolo11n_ArthroNat+flatbug.pt"
)

# Run inference on a single image
summary = run_inference(
    model=model,
    input_path="image.jpg",
    results_folder="results"
)

print(f"Total detections: {summary['total_detections']}")
```

### **With All Options**

```python
from run_model_with_cmd import load_model_from_huggingface, run_inference

# Load YOLO11l ONNX model
model = load_model_from_huggingface(
    repo_id="edgaremy/arthropod-detector",
    filename="yolo11l_ArthroNat+flatbug.onnx",
    verbose=True
)

# Run inference with all output options
summary = run_inference(
    model=model,
    input_path="images/",
    results_folder="results/yolo11l_test",
    save_crops=True,
    save_labels=True,
    save_bbox_view=True,
    conf_threshold=0.5,
    device="cuda",
    verbose=True
)

# Access summary statistics
print(f"Processed: {summary['processed_images']}/{summary['total_images']} images")
print(f"Images with detections: {summary['images_with_detections']}")
print(f"Images without detections: {summary['images_without_detections']}")
print(f"Total detections: {summary['total_detections']}")
print(f"Average detections/image: {summary['average_detections_per_image']:.2f}")
```

### **Model Selection Examples**

```python
# YOLO11n PyTorch (smallest, fastest)
model_n_pt = load_model_from_huggingface(
    repo_id="edgaremy/arthropod-detector",
    filename="yolo11n_ArthroNat+flatbug.pt"
)

# YOLO11l PyTorch (larger, more accurate)
model_l_pt = load_model_from_huggingface(
    repo_id="edgaremy/arthropod-detector",
    filename="yolo11l_ArthroNat+flatbug.pt"
)

# YOLO11n ONNX (optimized inference)
model_n_onnx = load_model_from_huggingface(
    repo_id="edgaremy/arthropod-detector",
    filename="yolo11n_ArthroNat+flatbug.onnx"
)

# YOLO11l ONNX (best accuracy, optimized)
model_l_onnx = load_model_from_huggingface(
    repo_id="edgaremy/arthropod-detector",
    filename="yolo11l_ArthroNat+flatbug.onnx"
)
```

### **Batch Processing Different Configurations**

```python
from run_model_with_cmd import load_model_from_huggingface, run_inference
from pathlib import Path

# Test different models on same dataset
configs = [
    {"filename": "yolo11n_ArthroNat+flatbug.pt", "name": "yolo11n_pt"},
    {"filename": "yolo11l_ArthroNat+flatbug.pt", "name": "yolo11l_pt"},
    {"filename": "yolo11n_ArthroNat+flatbug.onnx", "name": "yolo11n_onnx"},
    {"filename": "yolo11l_ArthroNat+flatbug.onnx", "name": "yolo11l_onnx"},
]

results = {}

for config in configs:
    print(f"\nTesting {config['name']}...")
    
    model = load_model_from_huggingface(
        repo_id="edgaremy/arthropod-detector",
        filename=config['filename'],
        verbose=False
    )
    
    summary = run_inference(
        model=model,
        input_path="test_images/",
        results_folder=f"results/{config['name']}",
        save_bbox_view=True,
        conf_threshold=0.25,
        device="cuda",
        verbose=False
    )
    
    results[config['name']] = summary

# Compare results
print("\n=== COMPARISON ===")
for name, summary in results.items():
    print(f"{name}: {summary['total_detections']} detections, "
          f"{summary['images_with_detections']} images with arthropods")
```

## 📁 Output Structure

```
results/
├── crop/                           # Cropped detections (if --save-crops)
│   ├── image1_0.jpg               # First detection from image1
│   ├── image1_1.jpg               # Second detection from image1
│   ├── image2_0.jpg               # First detection from image2
│   └── ...
│
├── labels/                         # YOLO format labels (if --save-labels)
│   ├── image1.txt                 # Format: class_id center_x center_y width height
│   ├── image2.txt                 # All values normalized [0-1]
│   ├── image3.txt                 # Empty file if no detections
│   └── ...
│
├── view-bbox/                      # Annotated images (if --save-bbox-view)
│   ├── image1.jpg                 # Original with bounding boxes drawn
│   ├── image2.jpg                 # Shows class name + confidence
│   ├── image3.jpg                 # Green boxes with labels
│   └── ...
│
└── summary.txt                     # Statistics summary (always created)
```

### **Example Label File Format** (`labels/image1.txt`)

```
0 0.512345 0.678901 0.123456 0.234567
0 0.234567 0.345678 0.098765 0.123456
1 0.789012 0.456789 0.156789 0.234567
```

Format: `class_id center_x center_y width height` (all normalized 0-1)

### **Example Summary File** (`summary.txt`)

```
============================================================
YOLO INFERENCE SUMMARY
============================================================
Date: 2025-11-20 15:30:45
Input path: images/
Results folder: results
Confidence threshold: 0.25
Device: cpu

STATISTICS:
------------------------------------------------------------
Total images: 150
Processed images: 150
Failed images: 0
Images with detections: 127
Images without detections: 23
Total detections (all bboxes): 342
Average detections per image: 2.28

OUTPUT OPTIONS:
------------------------------------------------------------
Save crops: True
  Crops folder: results/crop
Save labels: True
  Labels folder: results/labels
Save bbox views: True
  Bbox views folder: results/view-bbox
============================================================
```

---

## 🎯 Available Models

| Model | Format | Filename | Size | Speed | Accuracy |
|-------|--------|----------|------|-------|----------|
| YOLO11n | PyTorch | `yolo11n_ArthroNat+flatbug.pt` | ~6 MB | ⚡⚡⚡ | ⭐⭐ |
| YOLO11l | PyTorch | `yolo11l_ArthroNat+flatbug.pt` | ~50 MB | ⚡⚡ | ⭐⭐⭐ |
| YOLO11n | ONNX | `yolo11n_ArthroNat+flatbug.onnx` | ~10 MB | ⚡⚡⚡⚡ | ⭐⭐ |
| YOLO11l | ONNX | `yolo11l_ArthroNat+flatbug.onnx` | ~100 MB | ⚡⚡⚡ | ⭐⭐⭐ |

**Model Selection Guide:**
- **YOLO11n**: Fastest, good for real-time applications
- **YOLO11l**: More accurate, better for production use
- **PyTorch (.pt)**: Flexible, easier to fine-tune
- **ONNX (.onnx)**: Optimized inference, cross-platform deployment

---

## ⚙️ Arguments Reference

### **Positional Arguments**

| Argument | Description |
|----------|-------------|
| `input` | Path to image file or folder containing images |

### **Model Selection**

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--format` | `pt`, `onnx` | `pt` | Model format: PyTorch or ONNX |
| `--model-size` | `n`, `l` | `n` | Model size: n(ano) or l(arge) |

### **Output Options**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-r`, `--results-folder` | str | `results` | Output folder path |
| `--save-crops` | flag | False | Save cropped detections |
| `--save-labels` | flag | False | Save YOLO format labels |
| `--save-bbox-view` | flag | False | Save annotated images |

### **Inference Parameters**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--conf` | float | `0.25` | Confidence threshold (0.0-1.0) |
| `--device` | str | `cpu` | Device: cpu, cuda, 0, 1, etc. |

### **Advanced Options**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cache-dir` | str | None | Custom cache directory for models |
| `-v`, `--verbose` | flag | False | Enable verbose logging |
