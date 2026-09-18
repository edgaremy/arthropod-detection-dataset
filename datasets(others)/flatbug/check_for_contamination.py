"""Check for image contamination between our dataset (train/val/test) and the
flatbug source image folders.

Contamination is detected by comparing image *content* (perceptual hashes),
not filenames, so images that were renamed or re-encoded are still caught.

A 64-bit perceptual hash (pHash) is computed for every image via the
`imagehash` library. Two images are flagged when their hashes are within a
small Hamming distance (default 0 = perceptually identical).

Usage:
    python check_for_contamination.py
    python check_for_contamination.py --folders ArTaxOr Diopsis
    python check_for_contamination.py --threshold 5
    python check_for_contamination.py --dataset-root /path/to/dataset/images \
        --flatbug-root /path/to/flatbug-dataset
"""

import argparse
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

DEFAULT_FLATBUG_ROOT = "/media/eremy/451cfe78-3116-407a-bf7f-4f376a566e4e/flatbug-dataset"
DEFAULT_DATASET_ROOT = str(Path(__file__).resolve().parents[2] / "dataset" / "images")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
HASH_SIZE = 8           # 8x8 = 64-bit pHash
HIGHFREQ_FACTOR = 4     # imagehash default: resize to 32x32 before the DCT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check for image-content contamination between our dataset "
        "and the flatbug source image folders (perceptual-hash based)."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing the train/val/test image folders of our dataset.",
    )
    parser.add_argument(
        "--flatbug-root",
        type=str,
        default=DEFAULT_FLATBUG_ROOT,
        help="Root of the flatbug source image folders (one subfolder per source).",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=None,
        help="Optional subset of flatbug source folder names to check. "
        "If omitted, all subfolders are checked.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Maximum Hamming distance between perceptual hashes to count as a "
        "match. 0 = perceptually identical (default). Increase to also catch "
        "near-duplicates, at the cost of more false positives.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write the list of matches to (TSV).",
    )
    return parser.parse_args()


def phash(path: Path):
    """Compute a perceptual hash (imagehash.ImageHash) of the image at `path`."""
    with Image.open(path) as im:
        return imagehash.phash(im, hash_size=HASH_SIZE, highfreq_factor=HIGHFREQ_FACTOR)


def list_image_files(directory: Path):
    return [p for p in directory.iterdir() if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS]


def list_flatbug_folders(flatbug_root: Path, selected):
    if not flatbug_root.is_dir():
        raise SystemExit(f"Flatbug root not found: {flatbug_root}")
    all_folders = sorted(
        d for d in flatbug_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if selected:
        sel = set(selected)
        available = {d.name for d in all_folders}
        missing = sel - available
        if missing:
            raise SystemExit(
                "Requested flatbug folders not found: " + ", ".join(sorted(missing))
                + "\nAvailable: " + ", ".join(sorted(available))
            )
        all_folders = [d for d in all_folders if d.name in sel]
    return all_folders


def index_dataset(dataset_root: Path):
    """Return {hash: [(split, filename), ...]} and the number of images indexed
    (excluding unreadable ones)."""
    by_hash = defaultdict(list)
    total = 0
    skipped = 0
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        files = list_image_files(split_dir)
        for img_path in tqdm(files, desc=f"Indexing dataset/{split}", unit="img"):
            try:
                h = phash(img_path)
            except (UnidentifiedImageError, OSError, ValueError) as e:
                skipped += 1
                tqdm.write(f"  skip (unreadable): {img_path.name} ({e})")
                continue
            by_hash[h].append((split, img_path.name))
            total += 1
    return by_hash, total, skipped


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    flatbug_root = Path(args.flatbug_root)
    threshold = args.threshold

    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    print(f"Dataset root : {dataset_root}")
    print(f"Flatbug root : {flatbug_root}")
    print(f"Hamming threshold: {threshold}")
    if args.folders:
        print(f"Checking flatbug folders: {', '.join(args.folders)}")
    else:
        print("Checking all flatbug folders")

    by_hash, n_dataset, n_skipped = index_dataset(dataset_root)
    print(f"Indexed {n_dataset} dataset images ({len(by_hash)} unique hashes)"
          + (f", skipped {n_skipped} unreadable" if n_skipped else "")
          + "\n")

    folders = list_flatbug_folders(flatbug_root, args.folders)
    print(f"Scanning {len(folders)} flatbug folder(s)...\n")

    # For the near-duplicate path (threshold > 0) we need the list of dataset
    # hashes to compute Hamming distances against each flatbug image.
    dataset_hashes = list(by_hash.keys()) if threshold > 0 else None

    matches = []  # (flatbug_folder, flatbug_name, flatbug_hash, [(split, filename), ...])
    n_flatbug_imgs = 0
    n_flatbug_skipped = 0

    for folder in tqdm(folders, desc="Checking folders", unit="folder"):
        files = list_image_files(folder)
        for img_path in files:
            try:
                h = phash(img_path)
            except (UnidentifiedImageError, OSError, ValueError) as e:
                n_flatbug_skipped += 1
                tqdm.write(f"  skip (unreadable): {folder.name}/{img_path.name} ({e})")
                continue
            n_flatbug_imgs += 1

            if threshold == 0:
                entries = by_hash.get(h)
                if entries:
                    matches.append((folder.name, img_path.name, h, list(entries)))
            else:
                for ds_hash in dataset_hashes:
                    if h - ds_hash <= threshold:
                        matches.append(
                            (folder.name, img_path.name, h, list(by_hash[ds_hash]))
                        )

    # Report
    print("\n" + "=" * 72)
    print(f"Scanned {n_flatbug_imgs} flatbug images"
          + (f" ({n_flatbug_skipped} unreadable skipped)" if n_flatbug_skipped else "")
          + ".")
    print(f"Potential contaminations found: {len(matches)}")
    print("=" * 72)

    by_folder = defaultdict(list)
    for folder_name, fb_name, fb_hash, entries in matches:
        by_folder[folder_name].append((fb_name, fb_hash, entries))

    for folder_name in sorted(by_folder):
        rows = by_folder[folder_name]
        print(f"\n[{folder_name}] ({len(rows)} match(es))")
        for fb_name, fb_hash, entries in sorted(rows):
            splits = sorted({s for s, _ in entries})
            ds_files = "; ".join(f"{s}/{n}" for s, n in entries[:5])
            extra = f" (+{len(entries)-5} more)" if len(entries) > 5 else ""
            dist_str = "" if threshold == 0 else f" (dist<=threshold)"
            print(f"  {fb_name:<45} -> {'/'.join(splits)}{dist_str}")
            print(f"      dataset: {ds_files}{extra}")

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w") as f:
            f.write("flatbug_folder\tflatbug_filename\tflatbug_hash\t"
                    "dataset_splits\tdataset_filenames\n")
            for folder_name, fb_name, fb_hash, entries in sorted(matches):
                splits = ",".join(sorted({s for s, _ in entries}))
                files = ";".join(f"{s}/{n}" for s, n in entries)
                f.write(f"{folder_name}\t{fb_name}\t{fb_hash}\t{splits}\t{files}\n")
        print(f"\nWrote {len(matches)} matches to {out_path}")

    if matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
