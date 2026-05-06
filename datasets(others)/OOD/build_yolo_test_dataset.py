#!/usr/bin/env python3
"""Build a YOLO test dataset in OOD/ from COCO single-class annotations.

Output structure (at OOD root):
- images/test/
- labels/test/
- OOD.yaml
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path


# ============================
# Configuration (edit here)
# ============================
OOD_ROOT = Path(__file__).resolve().parent
COCO_JSON = OOD_ROOT / "annotations/cropped/processed/ground_truth_coco_single_cls.json"
SOURCE_IMAGES = OOD_ROOT / "cropped"
YAML_NAME = "OOD.yaml"

# Exactly one mode should be True.
COPY_IMAGES = True
SYMLINK_IMAGES = False

# If True, recreate images/test and labels/test when they already exist.
FORCE = True


def coco_bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / width
    y_center = (y + h / 2.0) / height
    w_norm = w / width
    h_norm = h / height
    return x_center, y_center, w_norm, h_norm


def safe_prepare_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Directory already exists: {path}. Use --force to overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if COPY_IMAGES == SYMLINK_IMAGES:
        raise ValueError("Set exactly one of COPY_IMAGES or SYMLINK_IMAGES to True")

    ood_root = OOD_ROOT.resolve()
    coco_json = COCO_JSON.resolve()
    source_images = SOURCE_IMAGES.resolve()

    if not coco_json.exists():
        raise FileNotFoundError(f"COCO JSON not found: {coco_json}")
    if not source_images.exists():
        raise FileNotFoundError(f"Source images directory not found: {source_images}")

    images_test_dir = ood_root / "images" / "test"
    labels_test_dir = ood_root / "labels" / "test"

    safe_prepare_dir(images_test_dir, force=FORCE)
    safe_prepare_dir(labels_test_dir, force=FORCE)

    with coco_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    if not images:
        raise ValueError("No images found in COCO JSON")

    # Single-class mapping for *_single_cls.json. If categories are present, keep name from id=1.
    class_name = "insect"
    for c in categories:
        if c.get("id") == 1 and isinstance(c.get("name"), str) and c["name"].strip():
            class_name = c["name"].strip()
            break

    anns_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_image_id[int(ann["image_id"])].append(ann)

    copied_or_linked = 0
    missing_images = 0
    label_files_written = 0

    for im in images:
        image_id = int(im["id"])
        width = int(im["width"])
        height = int(im["height"])
        file_name = im["file_name"]

        src_img = source_images / file_name
        dst_img = images_test_dir / file_name

        if not src_img.exists():
            missing_images += 1
            continue

        if SYMLINK_IMAGES:
            dst_img.symlink_to(src_img)
        else:
            shutil.copy2(src_img, dst_img)
        copied_or_linked += 1

        yolo_lines: list[str] = []
        for ann in anns_by_image_id.get(image_id, []):
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x_center, y_center, w_norm, h_norm = coco_bbox_to_yolo(bbox, width, height)
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_path = labels_test_dir / (Path(file_name).stem + ".txt")
        label_path.write_text("\n".join(yolo_lines), encoding="utf-8")
        label_files_written += 1

    yaml_path = ood_root / YAML_NAME
    yaml_content = (
        f"path: {ood_root}\n"
        "train: images/train/\n"
        "val: images/val/\n"
        "test: images/test/\n\n"
        "nc: 1\n\n"
        "names:\n"
        f"  0: \"{class_name}\"\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"YOLO dataset created at: {ood_root}")
    print(f"Images prepared in: {images_test_dir}")
    print(f"Labels written in: {labels_test_dir}")
    print(f"YAML written: {yaml_path}")
    print(f"Images processed: {copied_or_linked}")
    print(f"Label files written: {label_files_written}")
    print(f"Missing source images: {missing_images}")


if __name__ == "__main__":
    main()