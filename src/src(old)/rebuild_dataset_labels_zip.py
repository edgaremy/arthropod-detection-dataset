"""
Script 2: rebuild src/dataset_labels.zip so that each label file sits in the
split directory given by the new observation-level split.

Reads:  src/src(old)/dataset_labels.zip   (old archive: labels/{split}/*.txt)
        src/dataset_images.csv            (new official split, produced by script 1)
Writes: src/dataset_labels.zip

Each label file is named {taxon_id}_{photo_id}.txt and its new split is looked
up from the new CSV by (taxon_id, photo_id). Label files whose base name does
not appear in the CSV are kept in their original split directory and reported.
"""

import os
import zipfile
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_ZIP = os.path.join(SCRIPT_DIR, "dataset_labels.zip")
NEW_CSV = os.path.join(SCRIPT_DIR, os.pardir, "dataset_images.csv")
OUT_ZIP = os.path.join(SCRIPT_DIR, os.pardir, "dataset_labels.zip")

SPLITS = ("train", "val", "test")


def main():
    # New split for each label base name "{taxon_id}_{photo_id}".
    base_to_split = {}
    with open(NEW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            base_to_split[f"{row['taxon_id']}_{row['photo_id']}"] = row["split"]

    if os.path.exists(OUT_ZIP):
        os.remove(OUT_ZIP)

    moved = 0
    kept = 0
    unmatched = []
    per_split = {s: 0 for s in SPLITS}

    with zipfile.ZipFile(OLD_ZIP, "r") as zin, \
            zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zout:
        # Directory entries first, mirroring the original archive layout.
        zout.writestr(zipfile.ZipInfo("labels/"), "")
        for split in SPLITS:
            zout.writestr(zipfile.ZipInfo(f"labels/{split}/"), "")

        for info in zin.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/")
            # Expect labels/{split}/{base}.txt
            if len(parts) != 3 or parts[0] != "labels" or not parts[2].endswith(".txt"):
                print(f"[warn] Unexpected entry in old zip: {info.filename}")
                continue
            old_split, filename = parts[1], parts[2]
            base = filename[:-len(".txt")]

            new_split = base_to_split.get(base)
            if new_split is None:
                new_split = old_split
                unmatched.append(info.filename)
                kept += 1
            else:
                moved += 1
            per_split[new_split] += 1

            new_info = zipfile.ZipInfo(f"labels/{new_split}/{filename}",
                                       date_time=info.date_time)
            new_info.compress_type = zipfile.ZIP_DEFLATED
            new_info.external_attr = info.external_attr
            zout.writestr(new_info, zin.read(info.filename))

    print(f"Label files written: {moved + kept}")
    for split in SPLITS:
        print(f"  labels/{split}: {per_split[split]}")
    if unmatched:
        print(f"[warn] {kept} label(s) not found in the CSV, kept in original split:")
        for name in unmatched:
            print(f"  {name}")


if __name__ == "__main__":
    main()
