"""Create seeded 5x5 dataset visualization grids.

The script samples 5 random test images from each dataset column
(ArthroNat, flatbug, OOD, Lepinoc, SPIPOLL) and exports two PNG files:
1) images only
2) images with YOLO bounding boxes overlaid

Edit DATASET_SEEDS below to control deterministic sampling per dataset.
"""

from pathlib import Path
import json
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg


# User-configurable deterministic seeds (one per dataset column).
DATASET_SEEDS = {
	"ArthroNat": 8, #8
	"flatbug": 0,
	"OOD": 2,
	"Lepinoc": 83, #25, 83
	"SPIPOLL": 1,
}

N_ROWS = 5
N_COLS = 5
TEST_SPLIT = "test"
HEADER_FONT_SIZE = 30
OUTPUT_DPI = 180

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DATASETS = [
	("ArthroNat", REPO_ROOT / "dataset"),
	("flatbug", Path("datasets(others)/flatbug-yolo-split")),
	("SPIPOLL", REPO_ROOT / "datasets(others)" / "SPIPOLL"),
	("OOD", REPO_ROOT / "datasets(others)" / "OOD"),
	("Lepinoc", REPO_ROOT / "datasets(others)" / "Lepinoc"),
]

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


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
	x1 = (xc - w / 2.0) * img_w
	y1 = (yc - h / 2.0) * img_h
	x2 = (xc + w / 2.0) * img_w
	y2 = (yc + h / 2.0) * img_h
	return x1, y1, x2, y2


def load_yolo_boxes(label_path: Path, img_w: int, img_h: int) -> list[tuple[float, float, float, float]]:
	if not label_path.is_file():
		return []

	boxes = []
	with label_path.open("r", encoding="utf-8") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) != 5:
				continue

			try:
				_, xc, yc, bw, bh = map(float, parts)
			except ValueError:
				continue

			x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h)
			boxes.append((x1, y1, x2, y2))
	return boxes


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
		# Keep only calendar date so unique sampling is by day, not timestamp.
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
		# Split timeline into k segments and pick one date per segment.
		# This enforces distinct dates that are as far apart as possible.
		chosen_dates = []
		for i in range(k):
			start = (i * len(unique_dates)) // k
			end = ((i + 1) * len(unique_dates)) // k
			chosen_dates.append(unique_dates[rng.randrange(start, end)])
		chosen_paths = [rng.choice(paths_by_date[date_value]) for date_value in chosen_dates]
		print(f"OOD sampled distinct dates: {', '.join(sorted(chosen_dates))}")
		print("OOD selected images and dates:")
		for path in chosen_paths:
			print(f"  - {path.name}: {dates_by_filename.get(path.name, 'unknown')}")
		return chosen_paths

	# Not enough unique dates: keep strict one-image-per-date and pad with None.
	print(
		f"Warning: only {len(unique_dates)} unique OOD dates available in split '{TEST_SPLIT}'. "
		f"Padding {k - len(unique_dates)} slots with None to avoid repeated dates."
	)
	chosen_paths = [rng.choice(paths_by_date[date_value]) for date_value in unique_dates]
	chosen_paths.extend([None] * (k - len(chosen_paths)))

	print("OOD selected images and dates:")
	for path in chosen_paths:
		if path is None:
			print("  - None: no image")
		else:
			print(f"  - {path.name}: {dates_by_filename.get(path.name, 'unknown')}")

	return chosen_paths


def build_column_samples(dataset_seeds: dict[str, int]) -> list[tuple[str, Path, list[Path | None]]]:
	samples = []
	for dataset_name, dataset_root in DATASETS:
		rng = random.Random(dataset_seeds.get(dataset_name, 0))
		paths = list_test_images(dataset_root, split=TEST_SPLIT)
		if dataset_name == "OOD":
			chosen = sample_ood_paths_by_distinct_date(paths, N_ROWS, rng)
		else:
			chosen = sample_paths(paths, N_ROWS, rng)
		samples.append((dataset_name, dataset_root, chosen))
	return samples


def draw_grid(column_samples: list[tuple[str, Path, list[Path | None]]], output_path: Path, with_bboxes: bool) -> None:
	plt.rcParams["font.sans-serif"] = ["Nimbus Sans", "DejaVu Sans", "sans-serif"]
	plt.rcParams["font.family"] = "sans-serif"
	fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(16, 16), constrained_layout=False)

	for col, (dataset_name, dataset_root, image_list) in enumerate(column_samples):
		for row in range(N_ROWS):
			ax = axes[row, col]
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

			if with_bboxes:
				img_h, img_w = image.shape[0], image.shape[1]
				label_path = dataset_root / "labels" / TEST_SPLIT / f"{image_path.stem}.txt"
				boxes = load_yolo_boxes(label_path, img_w, img_h)

				for x1, y1, x2, y2 in boxes:
					rect = patches.Rectangle(
						(x1, y1),
						max(0.0, x2 - x1),
						max(0.0, y2 - y1),
						linewidth=1.5,
						edgecolor="#ff0040",
						facecolor="none",
					)
					ax.add_patch(rect)

	# Use figure-level headers so all dataset names share the exact same vertical position.
	for col, (dataset_name, _, _) in enumerate(column_samples):
		x_pos = (col + 0.5) / N_COLS
		fig.text(
			x_pos,
			0.985,
			dataset_name,
			ha="center",
			va="top",
			fontsize=HEADER_FONT_SIZE,
			fontweight="normal",
		)

	plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94, wspace=0.02, hspace=0.02)
	fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	column_samples = build_column_samples(DATASET_SEEDS)

	out_plain = SCRIPT_DIR / "dataset_grid_images.png"
	out_bbox = SCRIPT_DIR / "dataset_grid_with_bboxes.png"

	draw_grid(column_samples, out_plain, with_bboxes=False)
	draw_grid(column_samples, out_bbox, with_bboxes=True)

	print(f"Saved: {out_plain}")
	print(f"Saved: {out_bbox}")


if __name__ == "__main__":
	main()
