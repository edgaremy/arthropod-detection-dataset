from pathlib import Path

# Images found in common between the flatbug dataset and ours
# (see contamination_results.txt).
# 1ab637458816.jpg is excluded: it is a false positive.
CONTAMINATED_IMAGES = [
    "f97fc60acb7f.jpg",
    "f9c478422313.jpg",
    "fbe90aad5503.jpg",
    "ff661d2cc020.jpg",
]

yolo_split_dir = Path("/media/eremy/451cfe78-3116-407a-bf7f-4f376a566e4e/flatbug-yolo-split")

for name in CONTAMINATED_IMAGES:
    matches = list((yolo_split_dir / "images").rglob(name))
    if not matches:
        print(f"Not found (already removed?): {name}")
        continue
    for img_path in matches:
        split = img_path.parent.name
        lbl_path = yolo_split_dir / "labels" / split / (img_path.stem + ".txt")
        img_path.unlink()
        print(f"Deleted image: {img_path}")
        if lbl_path.exists():
            lbl_path.unlink()
            print(f"Deleted label: {lbl_path}")
