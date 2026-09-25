"""
Script 1: rebuild the official dataset split at the OBSERVATION level.

Reads:  src/src(old)/dataset_images.csv   (previous image-level split, kept as backup)
Writes: src/dataset_images.csv           (new official split)

Motivation: 315 observations contributed two photos to the dataset (batches 1 and 2
of the scraping process each requested one photo per taxon from overlapping
observations), and the old image-level split could place both photos of one
observation in different splits (189 observations spanned two splits). To remove
this contamination without deleting any image, the split is now assigned per
OBSERVATION: all photos of one observation land in the same split.

Method (per taxon, mirroring the per-species rules of the original
create_final_split.py; the random shuffle of split_dataset_detect.py is replaced
by a deterministic observation-uuid ordering so the result is reproducible):

  - the images of a taxon are grouped into units; a unit is one observation,
    except that rows sharing the same image file name (same taxon_id and
    photo_id, e.g. the exact-duplicate photo 55746_16314 listed under two
    observation uuids) are kept in the same unit as well
  - units are ordered by observation uuid (deterministic, no RNG)
  - if a taxon has a single unit, all its images go to train (a taxon whose
    only data is one observation cannot appear in two splits; train is chosen
    to keep test/val free of any train-observation contamination)
  - otherwise: test receives ceil(10% * n_images) images, val receives
    ceil(10% * n_images) images (0 when n_images == 2), assigned unit by unit
    in uuid order (a unit may overshoot its target by its extra photo);
    the remaining units go to train
"""

import csv
import math
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "dataset_images.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, os.pardir, "dataset_images.csv")

TEST_RATIO = 0.10
VAL_RATIO = 0.10


def image_filename(row):
    return f"{row['taxon_id']}_{row['photo_id']}.{row['extension']}"


def build_units(rows):
    """Group row indices into units (connected by observation uuid or file name)."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    by_observation = defaultdict(list)
    by_filename = defaultdict(list)
    for i, row in enumerate(rows):
        by_observation[row["observation_uuid"]].append(i)
        by_filename[image_filename(row)].append(i)
    for group in list(by_observation.values()) + list(by_filename.values()):
        for i, j in zip(group, group[1:]):
            union(i, j)

    units = defaultdict(list)
    for i in range(len(rows)):
        units[find(i)].append(i)
    # Sort key: the observation uuid(s) of the unit -> deterministic ordering.
    keyed = []
    for indices in units.values():
        uuids = ",".join(sorted({rows[i]["observation_uuid"] for i in indices}))
        keyed.append((uuids, indices))
    keyed.sort(key=lambda item: item[0])
    return [indices for _, indices in keyed]


def assign_splits(unit_sizes, n_images):
    """Greedy unit-by-unit assignment: test first, then val, then train."""
    test_nb = math.ceil(TEST_RATIO * n_images)
    val_nb = 0 if n_images == 2 else math.ceil(VAL_RATIO * n_images)

    splits = []
    phase = "test"
    cum_test = 0
    cum_val = 0
    for size in unit_sizes:
        if phase == "test" and cum_test < test_nb:
            splits.append("test")
            cum_test += size
            if cum_test >= test_nb:
                phase = "val"
        elif phase == "val" and cum_val < val_nb:
            splits.append("val")
            cum_val += size
            if cum_val >= val_nb:
                phase = "train"
        else:
            splits.append("train")
    return splits


def main():
    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    old_split = [row["split"] for row in rows]

    # Group rows per taxon, keeping the original row order.
    taxon_rows = defaultdict(list)
    for i, row in enumerate(rows):
        taxon_rows[row["taxon_id"]].append(i)

    new_split = [None] * len(rows)
    for taxon_id, indices in taxon_rows.items():
        taxon_data = [rows[i] for i in indices]
        units = build_units(taxon_data)
        if len(units) == 1:
            unit_splits = ["train"]
        else:
            unit_splits = assign_splits([len(u) for u in units], len(indices))
        for unit, split in zip(units, unit_splits):
            for i in unit:
                new_split[indices[i]] = split

    assert not any(s is None for s in new_split)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, split in zip(rows, new_split):
            row = dict(row)
            row["split"] = split
            writer.writerow(row)

    # Summary and verification
    changed = sum(1 for old, new in zip(old_split, new_split) if old != new)
    print(f"Rows written: {len(rows)}")
    print(f"Rows whose split changed: {changed}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {new_split.count(split)} (was {old_split.count(split)})")

    obs_split = defaultdict(set)
    for row, split in zip(rows, new_split):
        obs_split[row["observation_uuid"]].add(split)
    spanning = sum(1 for splits in obs_split.values() if len(splits) > 1)
    print(f"Observations spanning more than one split: {spanning} (expected 0)")
    if spanning:
        raise SystemExit("ERROR: observation contamination still present.")


if __name__ == "__main__":
    main()
