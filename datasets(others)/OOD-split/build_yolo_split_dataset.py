#!/usr/bin/env python3
"""Build a YOLO dataset in OOD-split/ from COCO single-class annotations.

Output structure (at OOD-split root):
- images/train/, images/val/, images/test/
- labels/train/, labels/val/, labels/test/
- OOD-split.yaml

Split strategy:
- Sort images by date_captured ascending (earliest first)
- Approximate 80/10/10 split over the sorted list
- Earliest images go to train, latest images go to test
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ============================
# Configuration (edit here)
# ============================
OOD_SPLIT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = OOD_SPLIT_ROOT.parent / "OOD"
COCO_JSON = SOURCE_ROOT / "annotations/cropped/processed/ground_truth_coco_single_cls.json"
SOURCE_IMAGES = SOURCE_ROOT / "cropped"
YAML_NAME = "OOD-split.yaml"

# Exactly one mode should be True.
COPY_IMAGES = True
SYMLINK_IMAGES = False

# If True, recreate images/* and labels/* when they already exist.
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
            raise FileExistsError(f"Directory already exists: {path}. Set FORCE=True to overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def split_counts(total: int) -> tuple[int, int, int]:
    """Return approximate 80/10/10 counts while keeping non-empty splits when possible."""
    train_n = int(total * 0.8)
    val_n = int(total * 0.1)
    test_n = total - train_n - val_n

    if total >= 3:
        if val_n == 0:
            val_n = 1
            train_n = max(1, train_n - 1)
        if test_n == 0:
            test_n = 1
            train_n = max(1, train_n - 1)

    # Ensure exact sum if any guard above adjusted counts.
    overflow = train_n + val_n + test_n - total
    if overflow > 0:
        train_n = max(0, train_n - overflow)

    return train_n, val_n, test_n


def main() -> None:
    if COPY_IMAGES == SYMLINK_IMAGES:
        raise ValueError("Set exactly one of COPY_IMAGES or SYMLINK_IMAGES to True")

    out_root = OOD_SPLIT_ROOT.resolve()
    coco_json = COCO_JSON.resolve()
    source_images = SOURCE_IMAGES.resolve()

    if not coco_json.exists():
        raise FileNotFoundError(f"COCO JSON not found: {coco_json}")
    if not source_images.exists():
        raise FileNotFoundError(f"Source images directory not found: {source_images}")

    image_dirs = {
        "train": out_root / "images" / "train",
        "val": out_root / "images" / "val",
        "test": out_root / "images" / "test",
    }
    label_dirs = {
        "train": out_root / "labels" / "train",
        "val": out_root / "labels" / "val",
        "test": out_root / "labels" / "test",
    }

    for split in ("train", "val", "test"):
        safe_prepare_dir(image_dirs[split], force=FORCE)
        safe_prepare_dir(label_dirs[split], force=FORCE)

    with coco_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    if not images:
        raise ValueError("No images found in COCO JSON")

    class_name = "insect"
    for c in categories:
        if c.get("id") == 1 and isinstance(c.get("name"), str) and c["name"].strip():
            class_name = c["name"].strip()
            break

    anns_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_image_id[int(ann["image_id"])].append(ann)

    # Deterministic ordering: earliest date first, then image id as tiebreaker.
    sorted_images = sorted(images, key=lambda im: (parse_date(im["date_captured"]), int(im["id"])))

    total_images = len(sorted_images)
    train_n, val_n, test_n = split_counts(total_images)

    split_by_image_id: dict[int, str] = {}
    for idx, im in enumerate(sorted_images):
        image_id = int(im["id"])
        if idx < train_n:
            split_by_image_id[image_id] = "train"
        elif idx < train_n + val_n:
            split_by_image_id[image_id] = "val"
        else:
            split_by_image_id[image_id] = "test"

    copied_or_linked = 0
    missing_images = 0
    label_files_written = 0
    per_split_count = {"train": 0, "val": 0, "test": 0}

    for im in sorted_images:
        image_id = int(im["id"])
        split = split_by_image_id[image_id]
        width = int(im["width"])
        height = int(im["height"])
        file_name = im["file_name"]

        src_img = source_images / file_name
        dst_img = image_dirs[split] / file_name

        if not src_img.exists():
            missing_images += 1
            continue

        if SYMLINK_IMAGES:
            dst_img.symlink_to(src_img)
        else:
            shutil.copy2(src_img, dst_img)
        copied_or_linked += 1
        per_split_count[split] += 1

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

        label_path = label_dirs[split] / (Path(file_name).stem + ".txt")
        label_path.write_text("\n".join(yolo_lines), encoding="utf-8")
        label_files_written += 1

    yaml_path = out_root / YAML_NAME
    yaml_content = (
        f"path: {out_root}\n"
        "train: images/train/\n"
        "val: images/val/\n"
        "test: images/test/\n\n"
        "nc: 1\n\n"
        "names:\n"
        f"  0: \"{class_name}\"\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"YOLO split dataset created at: {out_root}")
    print(f"YAML written: {yaml_path}")
    print(f"Total images in JSON: {total_images}")
    print(f"Target split counts (approx 80/10/10): train={train_n}, val={val_n}, test={test_n}")
    print(
        "Processed images by split: "
        f"train={per_split_count['train']}, val={per_split_count['val']}, test={per_split_count['test']}"
    )
    print(f"Images processed (copied/symlinked): {copied_or_linked}")
    print(f"Label files written: {label_files_written}")
    print(f"Missing source images: {missing_images}")


if __name__ == "__main__":
    main()
