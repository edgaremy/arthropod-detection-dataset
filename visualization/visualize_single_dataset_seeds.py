"""Interactively browse 5-image seed selections for one dataset.

Default behavior:
- Dataset: Lepinoc
- Starting seed: 1
- Right arrow: next seed (n + 1)
- Left arrow: previous seed (n - 1, clamped at 0)

The sampling behavior matches `visualization/visualize_datasets.py` exactly:
- Non-OOD datasets: `random.Random(seed).sample(sorted_test_images, 5)`
- OOD dataset: uses the same distinct-date sampling strategy based on COCO metadata.
"""

from pathlib import Path
import json
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

N_ROWS = 5
TEST_SPLIT = "test"
INITIAL_SEED = 1

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Select ONE dataset at a time by uncommenting exactly one block.
DATASET_NAME = "Lepinoc"
DATASET_ROOT = REPO_ROOT / "datasets(others)" / "Lepinoc"

# DATASET_NAME = "ArthroNat"
# DATASET_ROOT = REPO_ROOT / "dataset"

# DATASET_NAME = "flatbug"
# DATASET_ROOT = Path("datasets(others)/flatbug-yolo-split")

# DATASET_NAME = "SPIPOLL"
# DATASET_ROOT = REPO_ROOT / "datasets(others)" / "SPIPOLL"

# DATASET_NAME = "OOD"
# DATASET_ROOT = REPO_ROOT / "datasets(others)" / "OOD"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OOD_COCO_JSON = (
    REPO_ROOT
    / "datasets(others)"
    / "OOD"
    / "annotations"
    / "cropped"
    / "processed"
    / "ground_truth_coco_single_cls.json"
)


def list_test_images(dataset_root: Path, split: str = TEST_SPLIT) -> list[Path]:
    images_dir = dataset_root / "images" / split
    if not images_dir.is_dir():
        return []

    image_paths = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(image_paths)


def sample_paths(paths: list[Path], k: int, rng: random.Random) -> list[Path | None]:
    if len(paths) >= k:
        return rng.sample(paths, k)
    padded = list(paths)
    padded.extend([None] * (k - len(paths)))
    return padded


def load_ood_dates_by_filename(coco_json_path: Path) -> dict[str, str]:
    if not coco_json_path.is_file():
        return {}

    with coco_json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    dates_by_filename: dict[str, str] = {}
    for image_info in coco.get("images", []):
        file_name = image_info.get("file_name")
        date_captured = image_info.get("date_captured")
        if not isinstance(file_name, str) or not isinstance(date_captured, str):
            continue
        date_only = date_captured.strip()[:10]
        if not date_only:
            continue
        dates_by_filename[Path(file_name).name] = date_only

    return dates_by_filename


def sample_ood_paths_by_distinct_date(paths: list[Path], k: int, rng: random.Random) -> list[Path | None]:
    if len(paths) < k:
        padded = list(paths)
        padded.extend([None] * (k - len(paths)))
        return padded

    dates_by_filename = load_ood_dates_by_filename(OOD_COCO_JSON)
    if not dates_by_filename:
        return sample_paths(paths, k, rng)

    paths_by_date: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        date_value = dates_by_filename.get(path.name)
        if date_value:
            paths_by_date[date_value].append(path)

    if not paths_by_date:
        return sample_paths(paths, k, rng)

    unique_dates = sorted(paths_by_date.keys())
    if len(unique_dates) >= k:
        chosen_dates = []
        for i in range(k):
            start = (i * len(unique_dates)) // k
            end = ((i + 1) * len(unique_dates)) // k
            chosen_dates.append(unique_dates[rng.randrange(start, end)])
        return [rng.choice(paths_by_date[date_value]) for date_value in chosen_dates]

    chosen_paths = [rng.choice(paths_by_date[date_value]) for date_value in unique_dates]
    chosen_paths.extend([None] * (k - len(chosen_paths)))
    return chosen_paths


def sample_for_seed(dataset_name: str, dataset_root: Path, seed: int) -> list[Path | None]:
    rng = random.Random(seed)
    paths = list_test_images(dataset_root, split=TEST_SPLIT)

    if dataset_name == "OOD":
        return sample_ood_paths_by_distinct_date(paths, N_ROWS, rng)
    return sample_paths(paths, N_ROWS, rng)


def redraw_column(fig: plt.Figure, axes: list[plt.Axes], seed: int) -> None:
    image_list = sample_for_seed(DATASET_NAME, DATASET_ROOT, seed)

    for row in range(N_ROWS):
        ax = axes[row]
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])

        image_path = image_list[row]
        if image_path is None or not image_path.is_file():
            ax.text(0.5, 0.5, "No image", ha="center", va="center", fontsize=10)
            ax.set_frame_on(True)
            continue

        try:
            image = mpimg.imread(image_path)
        except Exception:
            ax.text(0.5, 0.5, "Read error", ha="center", va="center", fontsize=10)
            ax.set_frame_on(True)
            continue

        ax.imshow(image)
        ax.set_title(image_path.name, fontsize=10)

    fig.suptitle(
        f"{DATASET_NAME} | seed={seed} | right: n+1 | left: n-1",
        fontsize=14,
        y=0.995,
    )
    fig.canvas.draw_idle()


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans", "DejaVu Sans", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"

    fig, axes = plt.subplots(N_ROWS, 1, figsize=(5.5, 16), constrained_layout=False)
    if not isinstance(axes, list):
        axes = list(axes)

    state = {"seed": INITIAL_SEED}

    redraw_column(fig, axes, state["seed"])

    def on_key(event) -> None:
        if event.key == "right":
            state["seed"] += 1
            redraw_column(fig, axes, state["seed"])
        elif event.key == "left":
            state["seed"] = max(0, state["seed"] - 1)
            redraw_column(fig, axes, state["seed"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.01, top=0.97, hspace=0.25)
    plt.show()


if __name__ == "__main__":
    main()
