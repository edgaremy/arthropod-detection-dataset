"""
Script 3: reconcile the local dataset/ folder at the repo root with the new
observation-level split defined in src/dataset_images.csv.

For every image file found in dataset/images/{train,val,test}/ that is listed
in the CSV, the script checks whether its current split matches the CSV split.
Whenever it does not, the image is MOVED (cut/paste) to the correct split
directory, and its label file ({taxon_id}_{photo_id}.txt, searched in all
label split directories) is moved to the matching label split directory.

Files that are not listed in the CSV (e.g. leftover .tmp partial downloads)
are left in place and reported. Stale YOLO label cache files
(dataset/labels/*.cache) are deleted because they become invalid after the
move; ultralytics regenerates them on the next run.
"""

import csv
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
NEW_CSV = os.path.join(REPO_ROOT, "src", "dataset_images.csv")
IMAGES_DIR = os.path.join(REPO_ROOT, "dataset", "images")
LABELS_DIR = os.path.join(REPO_ROOT, "dataset", "labels")

SPLITS = ("train", "val", "test")


def main():
    filename_to_split = {}
    with open(NEW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            filename = f"{row['taxon_id']}_{row['photo_id']}.{row['extension']}"
            filename_to_split[filename] = row["split"]

    base_to_split = {}
    with open(NEW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            base_to_split[f"{row['taxon_id']}_{row['photo_id']}"] = row["split"]

    # Snapshot the image listings first so that files moved into a split that
    # is processed later are not scanned twice.
    listings = {}
    for split in SPLITS:
        split_dir = os.path.join(IMAGES_DIR, split)
        listings[split] = sorted(
            f for f in os.listdir(split_dir)
            if os.path.isfile(os.path.join(split_dir, f))
        ) if os.path.isdir(split_dir) else []

    def find_label(base):
        for split in SPLITS:
            path = os.path.join(LABELS_DIR, split, base + ".txt")
            if os.path.exists(path):
                return path
        return None

    images_moved = 0
    labels_moved = 0
    already_correct = 0
    unknown_files = []
    missing_labels = []
    seen = set()

    for split in SPLITS:
        for filename in listings[split]:
            path = os.path.join(IMAGES_DIR, split, filename)
            if filename in seen:
                print(f"[warn] {filename} found in more than one split directory")
            seen.add(filename)

            correct = filename_to_split.get(filename)
            if correct is None:
                unknown_files.append(os.path.join(split, filename))
                continue
            if correct == split:
                already_correct += 1
                continue

            # Move the image.
            shutil.move(path, os.path.join(IMAGES_DIR, correct, filename))
            images_moved += 1

            # Move its label along with it.
            base = os.path.splitext(filename)[0]
            label_path = find_label(base)
            if label_path is None:
                missing_labels.append(os.path.join(split, filename))
            else:
                shutil.move(label_path,
                            os.path.join(LABELS_DIR, correct, base + ".txt"))
                labels_moved += 1

    # Reconcile labels whose image is not on disk (e.g. failed downloads):
    # place them in the split given by the CSV, like the rebuilt zip does.
    orphan_labels_moved = 0
    for split in SPLITS:
        split_dir = os.path.join(LABELS_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        for filename in sorted(os.listdir(split_dir)):
            if not filename.endswith(".txt"):
                continue
            base = filename[:-len(".txt")]
            correct = base_to_split.get(base)
            if correct is not None and correct != split:
                shutil.move(os.path.join(split_dir, filename),
                            os.path.join(LABELS_DIR, correct, filename))
                orphan_labels_moved += 1

    # Remove stale YOLO cache files (regenerated automatically).
    stale_caches = []
    for entry in os.listdir(LABELS_DIR):
        if entry.endswith(".cache"):
            os.remove(os.path.join(LABELS_DIR, entry))
            stale_caches.append(entry)

    # Report CSV images that are not on disk (e.g. failed downloads).
    missing_images = sorted(set(filename_to_split) - seen)

    print(f"Images already in the correct split: {already_correct}")
    print(f"Images moved: {images_moved}")
    print(f"Labels moved: {labels_moved}")
    print(f"Orphan labels moved (image not on disk): {orphan_labels_moved}")
    if missing_labels:
        print(f"[warn] {len(missing_labels)} moved image(s) had no label file:")
        for name in missing_labels:
            print(f"  {name}")
    if unknown_files:
        print(f"[info] {len(unknown_files)} file(s) on disk are not in the CSV "
              f"(left in place):")
        for name in unknown_files[:20]:
            print(f"  {name}")
        if len(unknown_files) > 20:
            print(f"  ... and {len(unknown_files) - 20} more")
    if missing_images:
        print(f"[info] {len(missing_images)} CSV image(s) not found on disk "
              f"(failed or skipped downloads), e.g.:")
        for name in missing_images[:20]:
            print(f"  {name}")
        if len(missing_images) > 20:
            print(f"  ... and {len(missing_images) - 20} more")
    if stale_caches:
        print(f"[info] Removed {len(stale_caches)} stale YOLO label cache file(s): "
              f"{', '.join(stale_caches)}")


if __name__ == "__main__":
    main()
