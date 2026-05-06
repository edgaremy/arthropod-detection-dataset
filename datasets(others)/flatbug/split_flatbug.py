import shutil
import random
from pathlib import Path


def split_val_to_val_test(val_img_dir, val_lbl_dir, out_img_val, out_lbl_val, out_img_test, out_lbl_test):
    # Get set of image paths that are actually in the original val split
    original_val_imgs = set([p.resolve() for p in val_img_dir.rglob("*.*")])

    # Group files by source folder, but only include those in original val split
    sources = {}
    for img_path in original_val_imgs:
        source = img_path.parent.name
        sources.setdefault(source, []).append(img_path)

    for source, files in sources.items():
        random.shuffle(files)
        n_val = max(1, int(0.1 * len(files)))
        val_files = files[:n_val]
        test_files = files[n_val:]

        for img_file in val_files:
            rel_path = img_file.relative_to(val_img_dir)
            lbl_file = val_lbl_dir / rel_path.with_suffix('.txt')
            out_img = out_img_val / rel_path
            out_lbl = out_lbl_val / rel_path.with_suffix('.txt')
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, out_img)
            if lbl_file.exists():
                shutil.copy2(lbl_file, out_lbl)

        for img_file in test_files:
            rel_path = img_file.relative_to(val_img_dir)
            lbl_file = val_lbl_dir / rel_path.with_suffix('.txt')
            out_img = out_img_test / rel_path
            out_lbl = out_lbl_test / rel_path.with_suffix('.txt')
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, out_img)
            if lbl_file.exists():
                shutil.copy2(lbl_file, out_lbl)


input_dir = Path("datasets(others)/flatbug-yolo")
output_dir = Path("datasets(others)/flatbug-yolo-split")

# Prepare val/test output dirs
out_img_val = output_dir / 'images/val'
out_lbl_val = output_dir / 'labels/val'
out_img_test = output_dir / 'images/test'
out_lbl_test = output_dir / 'labels/test'

val_img_dir = input_dir / 'images/val'
val_lbl_dir = input_dir / 'labels/val'

split_val_to_val_test(val_img_dir, val_lbl_dir, out_img_val, out_lbl_val, out_img_test, out_lbl_test)
