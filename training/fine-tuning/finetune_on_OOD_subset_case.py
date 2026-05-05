#!/usr/bin/env python3
"""Run one OOD-split fine-tuning case for a given subset size and fold.

Examples:
- Transfer mode:
    python training/fine-tuning/finetune_on_OOD_subset_case.py --mode transfer --size 500 --fold 2
- Scratch mode:
    python training/fine-tuning/finetune_on_OOD_subset_case.py --mode scratch --size 500 --fold 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one OOD subset fine-tuning case")
    parser.add_argument("--mode", choices=["transfer", "scratch"], required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=float, default=0)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--freeze", type=int, default=10)

    parser.add_argument("--device", nargs="+", default=["0", "1"])
    parser.add_argument("--base-model", default="yolo11l.pt")
    parser.add_argument(
        "--transfer-weights",
        default="runs/arthro_and_flatbug/train/weights/best.pt",
    )

    parser.add_argument(
        "--ood-root",
        default="datasets(others)/OOD-split",
        help="Root folder that contains OOD-split.yaml and subsets/",
    )
    parser.add_argument(
        "--project-root",
        default="runs/fine_tuning",
        help="Base output folder under runs",
    )

    parser.add_argument(
        "--val-common-test",
        action="store_true",
        help="Also validate the trained model on OOD-split test split",
    )

    return parser.parse_args()


def build_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    subset_name = f"OOD-split{args.size}-fold{args.fold}"
    ood_root = Path(args.ood_root)
    dataset_yaml = ood_root / "subsets" / subset_name / f"{subset_name}.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Subset YAML not found: {dataset_yaml}")

    run_name = f"{args.mode}_11l_{subset_name}"
    project_dir = Path(args.project_root) / run_name

    return dataset_yaml, project_dir, run_name


def resolve_model(args: argparse.Namespace) -> YOLO:
    if args.mode == "transfer":
        return YOLO(args.transfer_weights)
    return YOLO(args.base_model)


def main() -> None:
    args = parse_args()

    if args.size <= 0:
        raise ValueError("--size must be > 0")
    if args.fold < 0:
        raise ValueError("--fold must be >= 0")

    dataset_yaml, project_dir, run_name = build_paths(args)

    print(f"mode={args.mode}")
    print(f"dataset={dataset_yaml}")
    print(f"run_name={run_name}")
    print(f"project={project_dir}")

    model = resolve_model(args)

    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        freeze=args.freeze,
        project=str(project_dir),
        name="train",
        device=args.device,
    )

    if args.val_common_test:
        best_weights = project_dir / "train" / "weights" / "best.pt"
        if not best_weights.exists():
            raise FileNotFoundError(f"Best weights not found after training: {best_weights}")

        model = YOLO(str(best_weights))
        model.val(
            data=str(Path(args.ood_root) / "OOD-split.yaml"),
            project=str(project_dir),
            name="val_common_test",
            device=args.device,
            split="test",
        )


if __name__ == "__main__":
    main()
