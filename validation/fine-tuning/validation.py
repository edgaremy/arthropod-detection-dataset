from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from ultralytics import YOLO


DATASET_ROOT = Path("datasets(others)/OOD-split")
OUTPUT_CSV = Path("validation/fine-tuning/finetuning_oodsplit_test_metrics.csv")

# DATASET_ROOT = Path("datasets(others)/Lepinoc-split")
# OUTPUT_CSV = Path("validation/fine-tuning/finetuning_lepinoc_test_metrics.csv")

CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5

SUBSET_SIZES = [100, 500, 1000, 2000]
NUM_FOLDS = 5


def _dataset_prefix(dataset_root: Path) -> str:
    """Return the split prefix used in fine-tuning run names."""
    return dataset_root.name


def _safe_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    intersection = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def _yolo_label_to_xyxy(label_line: str, im_width: int, im_height: int) -> np.ndarray:
    parts = label_line.strip().split()
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    x1 = (x_center - width / 2.0) * im_width
    y1 = (y_center - height / 2.0) * im_height
    x2 = (x_center + width / 2.0) * im_width
    y2 = (y_center + height / 2.0) * im_height
    return np.array([x1, y1, x2, y2], dtype=float)


def compute_image_metrics(
    label_lines: list[str],
    prediction_boxes_xyxy: np.ndarray,
    im_width: int,
    im_height: int,
    iou_threshold: float,
) -> dict[str, float]:
    n_labels = len(label_lines)
    n_preds = len(prediction_boxes_xyxy)

    bbox_relative_sizes: list[float] = []
    for line in label_lines:
        parts = line.strip().split()
        bbox_relative_sizes.append(float(parts[3]) * float(parts[4]))

    if n_labels == 0:
        return {
            "F1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mean_IoU": 0.0,
            "TP": 0,
            "FP": n_preds,
            "FN": 0,
            "avg_bbox_size": float(np.mean(bbox_relative_sizes)) if bbox_relative_sizes else 0.0,
        }

    label_boxes = np.stack([_yolo_label_to_xyxy(line, im_width, im_height) for line in label_lines], axis=0)

    if n_preds == 0:
        return {
            "F1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mean_IoU": 0.0,
            "TP": 0,
            "FP": 0,
            "FN": n_labels,
            "avg_bbox_size": float(np.mean(bbox_relative_sizes)) if bbox_relative_sizes else 0.0,
        }

    combinations = np.zeros((n_preds, n_labels), dtype=float)
    for i in range(n_preds):
        for j in range(n_labels):
            combinations[i, j] = _safe_iou(prediction_boxes_xyxy[i], label_boxes[j])

    row_ind, col_ind = linear_sum_assignment(-combinations)
    matched_ious = combinations[row_ind, col_ind]

    ious_with_zeros = np.zeros(n_labels, dtype=float)
    ious_with_zeros[col_ind] = matched_ious
    mean_iou = float(np.mean(ious_with_zeros))

    tp = int(np.sum(matched_ious >= iou_threshold))
    fp = int(n_preds - tp)
    fn = int(n_labels - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "F1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "mean_IoU": mean_iou,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "avg_bbox_size": float(np.mean(bbox_relative_sizes)) if bbox_relative_sizes else 0.0,
    }


def evaluate_model(
    model_path: Path,
    dataset_root: Path,
    split: str = "test",
    confidence: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    model = YOLO(str(model_path))

    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split

    all_f1: list[float] = []
    all_precision: list[float] = []
    all_recall: list[float] = []
    all_mean_iou: list[float] = []
    all_bbox_sizes: list[float] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    image_files = sorted([p for p in images_dir.iterdir() if p.is_file()])

    for img_path in tqdm(image_files, desc=f"Evaluating {model_path.name}"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        label_lines = label_path.read_text(encoding="utf-8").splitlines()
        if len(label_lines) == 0:
            continue

        results = model(str(img_path), save_txt=False, save=False, verbose=False, conf=confidence)
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()

        im_height, im_width = results[0].orig_shape[:2]
        metrics = compute_image_metrics(label_lines, pred_boxes, im_width, im_height, iou_threshold)

        all_f1.append(metrics["F1"])
        all_precision.append(metrics["precision"])
        all_recall.append(metrics["recall"])
        all_mean_iou.append(metrics["mean_IoU"])
        all_bbox_sizes.append(metrics["avg_bbox_size"])

        total_tp += int(metrics["TP"])
        total_fp += int(metrics["FP"])
        total_fn += int(metrics["FN"])

    if not all_f1:
        return {
            "avg_F1": 0.0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_mean_IoU": 0.0,
            "overall_F1": 0.0,
            "overall_precision": 0.0,
            "overall_recall": 0.0,
            "total_TP": 0,
            "total_FP": 0,
            "total_FN": 0,
            "avg_bbox_size": 0.0,
        }

    avg_f1 = float(np.mean(all_f1))
    avg_precision = float(np.mean(all_precision))
    avg_recall = float(np.mean(all_recall))
    avg_mean_iou = float(np.mean(all_mean_iou))
    avg_bbox_size = float(np.mean(all_bbox_sizes))

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    return {
        "avg_F1": avg_f1,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_mean_IoU": avg_mean_iou,
        "overall_F1": float(overall_f1),
        "overall_precision": float(overall_precision),
        "overall_recall": float(overall_recall),
        "total_TP": int(total_tp),
        "total_FP": int(total_fp),
        "total_FN": int(total_fn),
        "avg_bbox_size": avg_bbox_size,
    }


def build_models() -> list[tuple[str, str, int, int, Path]]:
    """Build the full validation matrix: baselines + all size/fold runs."""
    dataset_prefix = _dataset_prefix(DATASET_ROOT)

    models: list[tuple[str, str, int, int, Path]] = [
        (
            "arthro+flatbug",
            "baseline_arthro+flatbug",
            0,
            -1,
            Path("runs/arthro_and_flatbug/train/weights/best.pt"),
        ),
        (
            "fromscratch",
            "baseline_yolo11l",
            0,
            -1,
            Path("yolo11l.pt"),
        ),
    ]

    for n_images in SUBSET_SIZES:
        for fold in range(NUM_FOLDS):
            subset_name = f"{dataset_prefix}{n_images}-fold{fold}"
            models.append(
                (
                    "arthro+flatbug",
                    subset_name,
                    n_images,
                    fold,
                    Path(f"runs/fine_tuning/transfer_11l_{subset_name}/train/weights/best.pt"),
                )
            )
            models.append(
                (
                    "fromscratch",
                    subset_name,
                    n_images,
                    fold,
                    Path(f"runs/fine_tuning/scratch_11l_{subset_name}/train/weights/best.pt"),
                )
            )

    return models


def _sorted_models(
    models: Iterable[tuple[str, str, int, int, Path]],
) -> list[tuple[str, str, int, int, Path]]:
    return sorted(models, key=lambda x: (x[0], x[2], x[3], x[1]))


def main(models: list[tuple[str, str, int, int, Path]]) -> None:
    dataset_root = DATASET_ROOT
    output_csv = OUTPUT_CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not models:
        raise ValueError("The models list is empty. Add at least one model in __main__.")

    fieldnames = [
        "scenario",
        "model_name",
        "n_images",
        "fold",
        "model_path",
        "test_dataset",
        "avg_F1",
        "avg_precision",
        "avg_recall",
        "avg_mean_IoU",
        "overall_F1",
        "overall_precision",
        "overall_recall",
        "total_TP",
        "total_FP",
        "total_FN",
        "avg_bbox_size",
        "confidence",
        "IoU_threshold",
    ]

    missing_models: list[Path] = []

    with output_csv.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        for scenario, model_name, n_images, fold, model_path in _sorted_models(models):
            if not model_path.exists():
                missing_models.append(model_path)
                print(f"warning: checkpoint not found, skipping: {model_path}")
                continue

            print(
                f"Evaluating [{scenario}] {model_name} fold={fold} "
                f"({model_path}) on {dataset_root} [split=test]"
            )
            metrics = evaluate_model(
                model_path=model_path,
                dataset_root=dataset_root,
                split="test",
                confidence=CONFIDENCE,
                iou_threshold=IOU_THRESHOLD,
            )

            row = {
                "scenario": scenario,
                "model_name": model_name,
                "n_images": n_images,
                "fold": fold,
                "model_path": str(model_path),
                "test_dataset": str(dataset_root),
                "confidence": CONFIDENCE,
                "IoU_threshold": IOU_THRESHOLD,
            }
            row.update(metrics)
            writer.writerow(row)

    if missing_models:
        print(f"Finished with {len(missing_models)} missing checkpoints.")
    print(f"Done. Wrote metrics rows to: {output_csv}")


if __name__ == "__main__":
    main(build_models())
