#!/usr/bin/env python3
"""Build symlink-based fine-tuning subsets from OOD-split.

For each size N in SUBSET_SIZES, this script creates NUM_FOLDS subsets:
- subsets/OOD-split<N>-fold<F>/images/train
- subsets/OOD-split<N>-fold<F>/images/val
- subsets/OOD-split<N>-fold<F>/labels/train
- subsets/OOD-split<N>-fold<F>/labels/val
- subsets/OOD-split<N>-fold<F>/OOD-split<N>-fold<F>.yaml

Selection strategy (diversity-oriented):
- Read image date metadata from the source COCO JSON.
- For each split pool (train from OOD-split train, val from OOD-split val),
  sort by date and pick temporally spread images for each fold.
- Enforce disjoint folds when feasible; otherwise warn and allow overlap.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


# ============================
# Configuration (edit here)
# ============================
SUBSET_SIZES = [100, 500, 1000, 2000]
VAL_RATIO = 0.20
NUM_FOLDS = 5

OOD_SPLIT_ROOT = Path(__file__).resolve().parent
SUBSETS_ROOT = OOD_SPLIT_ROOT / "subsets"

SOURCE_ROOT = OOD_SPLIT_ROOT.parent / "OOD"
COCO_JSON = SOURCE_ROOT / "annotations/cropped/processed/ground_truth_coco_single_cls.json"

FORCE = True


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def safe_prepare_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Directory already exists: {path}. Set FORCE=True to overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clamp(n: int, low: int, high: int) -> int:
    return max(low, min(high, n))


def _nearest_unused_index(target: float, used: set[int], n: int) -> int:
    start = int(round(target))
    start = clamp(start, 0, n - 1)
    if start not in used:
        return start

    offset = 1
    while True:
        left = start - offset
        right = start + offset
        left_ok = left >= 0 and left not in used
        right_ok = right < n and right not in used

        if left_ok and right_ok:
            if abs(left - target) <= abs(right - target):
                return left
            return right
        if left_ok:
            return left
        if right_ok:
            return right
        offset += 1


def select_evenly_spread_indices(total: int, k: int) -> list[int]:
    """Pick k indices from range(total) as spread out as possible.

    The output is deterministic and sorted.
    """
    if total <= 0 or k <= 0:
        return []
    if k >= total:
        return list(range(total))
    if k == 1:
        return [total // 2]

    used: set[int] = set()
    selected: list[int] = []
    step = (total - 1) / (k - 1)

    for i in range(k):
        target = i * step
        idx = _nearest_unused_index(target, used, total)
        used.add(idx)
        selected.append(idx)

    selected.sort()
    return selected


def select_fold_indices(
    total: int,
    k: int,
    fold_idx: int,
    num_folds: int,
) -> tuple[list[int], bool]:
    """Select fold-specific indices from a sorted list.

    Returns (indices, overlap_used). When disjoint allocation is feasible,
    this uses a stride pattern that is disjoint across folds. Otherwise, it
    falls back to a deterministic shifted evenly-spread pattern.
    """
    if total <= 0 or k <= 0:
        return [], False

    k = min(k, total)

    if total >= k * num_folds:
        # Important: spread k*num_folds anchors across the full timeline first,
        # then assign every num_folds-th anchor to each fold.
        anchors = select_evenly_spread_indices(total, k * num_folds)
        selected = anchors[fold_idx::num_folds][:k]
        return selected, False

    base = select_evenly_spread_indices(total, k)
    shift = int(round((fold_idx * total) / num_folds)) % total
    selected = [((idx + shift) % total) for idx in base]
    selected.sort()
    return selected, True


def choose_time_diverse_files(
    file_names: list[str],
    date_by_file: dict[str, datetime],
    k: int,
) -> list[str]:
    with_dates: list[tuple[datetime, str]] = []
    missing_dates: list[str] = []

    for fn in file_names:
        dt = date_by_file.get(fn)
        if dt is None:
            missing_dates.append(fn)
        else:
            with_dates.append((dt, fn))

    with_dates.sort(key=lambda x: (x[0], x[1]))

    # Prefer date-aware samples first, and only backfill with undated files if needed.
    target = min(k, len(file_names))
    pick_from_dated = min(target, len(with_dates))
    idxs = select_evenly_spread_indices(len(with_dates), pick_from_dated)
    selected = [with_dates[i][1] for i in idxs]

    if len(selected) < target:
        missing_dates.sort()
        need = target - len(selected)
        selected.extend(missing_dates[:need])

    return selected


def choose_time_diverse_files_for_fold(
    file_names: list[str],
    date_by_file: dict[str, datetime],
    k: int,
    fold_idx: int,
    num_folds: int,
) -> tuple[list[str], bool]:
    with_dates: list[tuple[datetime, str]] = []
    missing_dates: list[str] = []

    for fn in file_names:
        dt = date_by_file.get(fn)
        if dt is None:
            missing_dates.append(fn)
        else:
            with_dates.append((dt, fn))

    with_dates.sort(key=lambda x: (x[0], x[1]))
    dated_files = [fn for _, fn in with_dates]
    missing_dates.sort()

    target = min(k, len(file_names))
    pick_from_dated = min(target, len(dated_files))

    dated_idxs, dated_overlap = select_fold_indices(
        total=len(dated_files),
        k=pick_from_dated,
        fold_idx=fold_idx,
        num_folds=num_folds,
    )
    selected = [dated_files[i] for i in dated_idxs]

    overlap_used = dated_overlap

    if len(selected) < target:
        need = target - len(selected)
        missing_idxs, missing_overlap = select_fold_indices(
            total=len(missing_dates),
            k=need,
            fold_idx=fold_idx,
            num_folds=num_folds,
        )
        selected.extend(missing_dates[i] for i in missing_idxs)
        overlap_used = overlap_used or missing_overlap

    return selected, overlap_used


def symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def date_coverage_summary(file_names: list[str], date_by_file: dict[str, datetime]) -> str:
    dates = sorted(date_by_file[fn] for fn in file_names if fn in date_by_file)
    if not dates:
        return "no dated samples"
    return (
        f"dated_samples={len(dates)}, "
        f"date_min={dates[0].date()}, date_max={dates[-1].date()}"
    )


def load_date_by_file(coco_json: Path) -> tuple[dict[str, datetime], str]:
    with coco_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    categories = coco.get("categories", [])

    date_by_file: dict[str, datetime] = {}
    for im in images:
        file_name = im.get("file_name")
        date_str = im.get("date_captured")
        if not isinstance(file_name, str) or not file_name:
            continue
        if not isinstance(date_str, str) or not date_str:
            continue
        try:
            date_by_file[file_name] = parse_date(date_str)
        except ValueError:
            continue

    class_name = "insect"
    for c in categories:
        if c.get("id") == 1 and isinstance(c.get("name"), str) and c["name"].strip():
            class_name = c["name"].strip()
            break

    return date_by_file, class_name


def list_split_files(images_dir: Path, labels_dir: Path) -> list[str]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

    files = []
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        label = labels_dir / (p.stem + ".txt")
        if label.exists():
            files.append(p.name)

    files.sort()
    return files


def write_yaml(dataset_root: Path, yaml_name: str, class_name: str) -> Path:
    yaml_path = dataset_root / yaml_name
    yaml_content = (
        f"path: {dataset_root}\n"
        "train: images/train/\n"
        "val: images/val/\n\n"
        "nc: 1\n\n"
        "names:\n"
        f"  0: \"{class_name}\"\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def build_one_subset(
    n_train_target: int,
    train_pool: list[str],
    val_pool: list[str],
    date_by_file: dict[str, datetime],
    class_name: str,
    fold_idx: int,
    num_folds: int,
) -> tuple[bool, bool]:
    n_train = min(n_train_target, len(train_pool))
    n_val_target = int(round(n_train_target * VAL_RATIO))
    if n_train_target > 0 and n_val_target == 0:
        n_val_target = 1
    n_val = min(n_val_target, len(val_pool))

    subset_name = f"OOD-split{n_train_target}-fold{fold_idx}"
    subset_root = SUBSETS_ROOT / subset_name

    out_img_train = subset_root / "images" / "train"
    out_img_val = subset_root / "images" / "val"
    out_lbl_train = subset_root / "labels" / "train"
    out_lbl_val = subset_root / "labels" / "val"

    safe_prepare_dir(out_img_train, force=FORCE)
    safe_prepare_dir(out_img_val, force=FORCE)
    safe_prepare_dir(out_lbl_train, force=FORCE)
    safe_prepare_dir(out_lbl_val, force=FORCE)

    chosen_train, train_overlap_used = choose_time_diverse_files_for_fold(
        train_pool,
        date_by_file,
        n_train,
        fold_idx,
        num_folds,
    )
    chosen_val, val_overlap_used = choose_time_diverse_files_for_fold(
        val_pool,
        date_by_file,
        n_val,
        fold_idx,
        num_folds,
    )

    src_img_train = OOD_SPLIT_ROOT / "images" / "train"
    src_img_val = OOD_SPLIT_ROOT / "images" / "val"
    src_lbl_train = OOD_SPLIT_ROOT / "labels" / "train"
    src_lbl_val = OOD_SPLIT_ROOT / "labels" / "val"

    for fn in chosen_train:
        symlink_file(src_img_train / fn, out_img_train / fn)
        symlink_file(src_lbl_train / (Path(fn).stem + ".txt"), out_lbl_train / (Path(fn).stem + ".txt"))

    for fn in chosen_val:
        symlink_file(src_img_val / fn, out_img_val / fn)
        symlink_file(src_lbl_val / (Path(fn).stem + ".txt"), out_lbl_val / (Path(fn).stem + ".txt"))

    yaml_path = write_yaml(subset_root, f"{subset_name}.yaml", class_name)

    print(f"[{subset_name}] created: {subset_root}")
    print(f"  train requested={n_train_target}, selected={len(chosen_train)}")
    print(f"  val requested={n_val_target}, selected={len(chosen_val)}")
    print(f"  train coverage: {date_coverage_summary(chosen_train, date_by_file)}")
    print(f"  val coverage: {date_coverage_summary(chosen_val, date_by_file)}")
    if train_overlap_used:
        print("  warning: train fold overlap used (insufficient pool for strict disjointness)")
    if val_overlap_used:
        print("  warning: val fold overlap used (insufficient pool for strict disjointness)")
    print(f"  yaml: {yaml_path}")

    return train_overlap_used, val_overlap_used


def main() -> None:
    if not COCO_JSON.exists():
        raise FileNotFoundError(f"COCO JSON not found: {COCO_JSON}")

    src_train_images = OOD_SPLIT_ROOT / "images" / "train"
    src_val_images = OOD_SPLIT_ROOT / "images" / "val"
    src_train_labels = OOD_SPLIT_ROOT / "labels" / "train"
    src_val_labels = OOD_SPLIT_ROOT / "labels" / "val"

    train_pool = list_split_files(src_train_images, src_train_labels)
    val_pool = list_split_files(src_val_images, src_val_labels)

    if not train_pool:
        raise ValueError("No train images found in OOD-split")
    if not val_pool:
        raise ValueError("No val images found in OOD-split")

    SUBSETS_ROOT.mkdir(parents=True, exist_ok=True)

    date_by_file, class_name = load_date_by_file(COCO_JSON)

    print("Building fine-tuning subsets from OOD-split")
    print(f"Train pool size: {len(train_pool)}")
    print(f"Val pool size: {len(val_pool)}")
    print(f"Subset sizes: {SUBSET_SIZES}")
    print(f"Val ratio: {VAL_RATIO}")
    print(f"Num folds: {NUM_FOLDS}")

    for n in SUBSET_SIZES:
        if not isinstance(n, int) or n <= 0:
            print(f"Skipping invalid subset size: {n}")
            continue

        n_train = min(n, len(train_pool))
        n_val_target = int(round(n * VAL_RATIO))
        if n > 0 and n_val_target == 0:
            n_val_target = 1
        n_val = min(n_val_target, len(val_pool))

        if n_train * NUM_FOLDS > len(train_pool):
            print(
                "warning: strict train-fold disjointness is not possible "
                f"for size {n} ({NUM_FOLDS}x{n_train} > pool {len(train_pool)})."
            )
        if n_val * NUM_FOLDS > len(val_pool):
            print(
                "warning: strict val-fold disjointness is not possible "
                f"for size {n} ({NUM_FOLDS}x{n_val} > pool {len(val_pool)})."
            )

        size_train_overlap = False
        size_val_overlap = False
        for fold_idx in range(NUM_FOLDS):
            train_overlap, val_overlap = build_one_subset(
                n,
                train_pool,
                val_pool,
                date_by_file,
                class_name,
                fold_idx,
                NUM_FOLDS,
            )
            size_train_overlap = size_train_overlap or train_overlap
            size_val_overlap = size_val_overlap or val_overlap

        print(
            f"size {n} summary: train_overlap_used={size_train_overlap}, "
            f"val_overlap_used={size_val_overlap}"
        )

    print("All done.")


if __name__ == "__main__":
    main()
