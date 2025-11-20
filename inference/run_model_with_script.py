#!/usr/bin/env python3
"""
Simple arthropod detection script using run_model_with_cmd.py module.
"""

from run_model_with_cmd import load_model_from_huggingface, run_inference

# Load model
model = load_model_from_huggingface(
    repo_id="edgaremy/arthropod-detector",
    filename="yolo11l_ArthroNat+flatbug.onnx"
)

# Run inference
summary = run_inference(
    model=model,
    input_path="images/",
    results_folder="results/batch_processing",
    save_crops=True,
    save_labels=True,
    save_bbox_view=True,
    conf_threshold=0.5,
    device="cpu",
    verbose=True
)
