from pathlib import Path
import csv


BATCH_ROOT_FOLDER = "/media/disk2/Arthropods/"
OUTPUT_CSV = Path(__file__).with_name("batch_stats.csv")
WAVE_START = 1
WAVE_END = 22

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def count_files(folder: Path, allowed_extensions: set[str] | None = None) -> int:
    if not folder.exists():
        return 0

    total = 0
    for subfolder in ("train", "val"):
        split_dir = folder / subfolder
        if not split_dir.exists():
            continue

        for path in split_dir.rglob("*"):
            if not path.is_file():
                continue
            if allowed_extensions is None or path.suffix.lower() in allowed_extensions:
                total += 1

    return total


def main() -> None:
    root = Path(BATCH_ROOT_FOLDER)

    rows = []
    previous_total_images = 0
    previous_total_images_with_labels = 0

    for wave_id in range(WAVE_START, WAVE_END + 1):
        wave_name = f"LIMIT{wave_id}"
        wave_folder = root / wave_name / "dataset"

        images_folder = wave_folder / "images"
        labels_folder = wave_folder / "labels"

        current_total_images = count_files(images_folder, IMAGE_EXTENSIONS)
        current_total_labels = count_files(labels_folder, {".txt"})

        incremental_images = current_total_images - previous_total_images
        incremental_labels = current_total_labels - previous_total_images_with_labels

        rows.append(
            {
                "wave": wave_name,
                "images": incremental_images,
                "images_with_labels": incremental_labels,
                "total_images": current_total_images,
                "total_images_with_labels": current_total_labels,
            }
        )

        previous_total_images = current_total_images
        previous_total_images_with_labels = current_total_labels

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "wave",
                "images",
                "images_with_labels",
                "total_images",
                "total_images_with_labels",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
