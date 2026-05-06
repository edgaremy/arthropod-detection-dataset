import os
import sys
import json
import shutil
from pathlib import Path
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

def parse_args():
    parser = ArgumentParser(description="Convert flatbug segmentation dataset to YOLO detection format.")
    parser.add_argument("input_dir", type=str, help="Path to flatbug-dataset root directory")
    parser.add_argument("output_dir", type=str, help="Path to output YOLO dataset directory")
    return parser.parse_args()

def load_metadata(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def convert_segmentation_to_bbox(segmentation):
    # segmentation: [x1, y1, x2, y2, x3, y3, ...] or [[...], [...], ...]
    if not segmentation:
        return None  # Invalid segmentation
    # If segmentation is a list of lists (COCO format), flatten it
    if isinstance(segmentation[0], list):
        segmentation = [coord for sublist in segmentation for coord in sublist]
    if not segmentation or len(segmentation) < 4:
        return None  # Not enough points
    xs = segmentation[0::2]
    ys = segmentation[1::2]
    if not xs or not ys:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max, y_max

def bbox_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def convert_flatbug_to_yolo(input_dir, output_dir):
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    # Collect all classes (COCO format: categories is a list of dicts with id and name)
    class_names = set()
    image_infos = []

    for source_dir in input_dir.iterdir():
        if not source_dir.is_dir():
            continue
        json_files = list(source_dir.glob("*.json"))
        if not json_files:
            continue
        meta = load_metadata(json_files[0])
        # Build category id to name mapping
        if "categories" in meta:
            catid2name = {cat["id"]: cat["name"] for cat in meta["categories"]}
        else:
            catid2name = {}
        for img_info in meta["images"]:
            img_path = source_dir / img_info["file_name"]
            annots = [a for a in meta["annotations"] if a["image_id"] == img_info["id"]]
            # Add category names from annotations
            for a in annots:
                if "category_id" in a and a["category_id"] in catid2name:
                    class_names.add(catid2name[a["category_id"]])
            image_infos.append({
                "img_path": img_path,
                "width": img_info["width"],
                "height": img_info["height"],
                "annotations": annots,
                "split": img_info.get("split", "train"),
                "catid2name": catid2name
            })

    class_names = sorted(list(class_names))
    class2idx = {name: idx for idx, name in enumerate(class_names)}

    # Split images by split field or fallback to random split
    splits = {"train": [], "val": [], "test": []}
    for info in image_infos:
        split = info["split"]
        if split not in splits:
            split = "train"
        splits[split].append(info)

    # If no val/test, split train
    if not splits["val"] and not splits["test"]:
        train_imgs, val_imgs = train_test_split(splits["train"], test_size=0.2, random_state=42)
        splits["train"] = train_imgs
        splits["val"] = val_imgs

    # Copy images and write labels
    for split in ["train", "val", "test"]:
        split_clean = split.strip()
        if not splits[split]:
            continue
        split_img_dir = images_dir / split_clean
        split_lbl_dir = labels_dir / split_clean
        ensure_dir(split_img_dir)
        ensure_dir(split_lbl_dir)
        for info in splits[split]:
            img_dst = split_img_dir / info["img_path"].name
            if not info["img_path"].exists():
                print(f"Warning: image file not found, skipping: {info['img_path']}")
                continue
            shutil.copy2(info["img_path"], img_dst)
            label_path = split_lbl_dir / (info["img_path"].stem + ".txt")
            with open(label_path, "w") as f:
                for ann in info["annotations"]:
                    # Use COCO format: category_id and catid2name
                    if "category_id" in ann and ann["category_id"] in info["catid2name"]:
                        class_name = info["catid2name"][ann["category_id"]]
                        class_idx = class2idx[class_name]
                    else:
                        continue
                    bbox = None
                    if "segmentation" in ann:
                        bbox = convert_segmentation_to_bbox(ann["segmentation"])
                    elif "bbox" in ann:
                        bbox = ann["bbox"]
                    if bbox is None:
                        continue  # Skip invalid or empty segmentations
                    yolo_bbox = bbox_to_yolo(bbox, info["width"], info["height"])
                    f.write(f"{class_idx} {' '.join(f'{v:.6f}' for v in yolo_bbox)}\n")

    # Write data.yaml
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        if splits["test"]:
            f.write(f"test: images/test\n")
        f.write(f"names: {class_names}\n")

# Example usage:
input_dir = Path("datasets(others)/flatbug/flatbug-dataset")
output_dir = Path("datasets(others)/flatbug-yolo")
convert_flatbug_to_yolo(input_dir, output_dir)