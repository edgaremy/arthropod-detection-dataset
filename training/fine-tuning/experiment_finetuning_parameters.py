from pathlib import Path

from ultralytics import YOLO


# Keep this script intentionally simple, like training/train_with_flatbug.py.
model = YOLO("runs/arthro_and_flatbug/train/weights/best.pt")
# model = YOLO("yolo11l.pt")  # Uncomment for scratch mode

# model.train(
# 	data="datasets(others)/Lepinoc-split/subsets/Lepinoc-split1000-fold0/Lepinoc-split1000-fold0.yaml",
# 	epochs=20,
# 	warmup_epochs=0,
#     optimizer="AdamW",
# 	lr0=0.001,
# 	lrf=0.01,
# 	freeze=10,
# 	project="runs/fine_tuning/experiment",
# 	name="train",
# 	device=["0", "1"],
# )

# # Validate on Lepinoc test split.
# best_weights = Path("runs/fine_tuning/experiment") / "train" / "weights" / "best.pt"
# model = YOLO(str(best_weights))
# model.val(
# 	data="datasets(others)/Lepinoc-split/Lepinoc-split.yaml",
# 	project="runs/fine_tuning/experiment",
# 	name="val_common_test",
# 	device=["0", "1"],
# 	split="test",
# )

model.train(
	data="datasets(others)/OOD-split/subsets/OOD-split500-fold0/OOD-split500-fold0.yaml",
	epochs=20,
	warmup_epochs=0,
    optimizer="AdamW",
	lr0=0.001,
	lrf=0.01,
	freeze=10,
	project="runs/fine_tuning/experiment",
	name="train3",
	device=["0", "1"],
)

# Validate on OOD test split.
best_weights = Path("runs/fine_tuning/experiment") / "train3" / "weights" / "best.pt"
model = YOLO(str(best_weights))
model.val(
	data="datasets(others)/OOD-split/OOD-split.yaml",
	project="runs/fine_tuning/experiment",
	name="val_common_test3",
	device=["0", "1"],
	split="test",
)
