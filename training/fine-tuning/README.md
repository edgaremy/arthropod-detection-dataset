# Fine-Tuning Scripts (OOD-split + Lepinoc-split)

This folder provides one-case runners and matrix launchers for 5-fold subset
fine-tuning experiments.

## Datasets

- OOD:
	- `datasets(others)/OOD-split/OOD-split.yaml`
	- `datasets(others)/OOD-split/subsets/OOD-split<N>-fold<F>/OOD-split<N>-fold<F>.yaml`
- Lepinoc:
	- `datasets(others)/Lepinoc-split/Lepinoc-split.yaml`
	- `datasets(others)/Lepinoc-split/subsets/Lepinoc-split<N>-fold<F>/Lepinoc-split<N>-fold<F>.yaml`

Defaults assume two GPUs: `DEVICES="0 1"`.

## Scripts

- OOD single-case runner: `finetune_on_OOD_subset_case.py`
- OOD matrix launcher: `run_matrix_ood_folds.sh`
- OOD wrappers:
	- `run_all_arthro_flatbug_11l_OODsplit.sh` (transfer)
	- `run_all_fromscratch_11l_OODsplit.sh` (scratch)

- Lepinoc single-case runner: `finetune_on_Lepinoc_subset_case.py`
- Lepinoc matrix launcher: `run_matrix_lepinoc_folds.sh`
- Lepinoc wrappers:
	- `run_all_arthro_flatbug_11l_Lepinocsplit.sh` (transfer)
	- `run_all_fromscratch_11l_Lepinocsplit.sh` (scratch)

All matrix launchers keep training hyperparameters fixed across subset sizes and
folds, so size/fold is the main variable.

## Quick Usage

Run from repository root.

OOD one-case:

```bash
python training/fine-tuning/finetune_on_OOD_subset_case.py --mode transfer --size 500 --fold 2
python training/fine-tuning/finetune_on_OOD_subset_case.py --mode scratch --size 500 --fold 2
```

Lepinoc one-case:

```bash
python training/fine-tuning/finetune_on_Lepinoc_subset_case.py --mode transfer --size 500 --fold 2
python training/fine-tuning/finetune_on_Lepinoc_subset_case.py --mode scratch --size 500 --fold 2
```

OOD matrix:

```bash
bash training/fine-tuning/run_matrix_ood_folds.sh
```

Lepinoc matrix:

```bash
bash training/fine-tuning/run_matrix_lepinoc_folds.sh
```

Transfer-only or scratch-only (works for either matrix launcher):

```bash
MODES="transfer" bash training/fine-tuning/run_matrix_ood_folds.sh
MODES="scratch" bash training/fine-tuning/run_matrix_lepinoc_folds.sh
```

Filtered matrix (example):

```bash
MODES="transfer scratch" SIZES="100 500" FOLDS="0 1" bash training/fine-tuning/run_matrix_lepinoc_folds.sh
```

Dry run and shared overrides:

```bash
DRY_RUN=1 bash training/fine-tuning/run_matrix_ood_folds.sh
EPOCHS=20 WARMUP_EPOCHS=0 LR0=0.001 LRF=0.01 FREEZE=10 DEVICES="0 1" VAL_COMMON_TEST=1 \
	bash training/fine-tuning/run_matrix_lepinoc_folds.sh
```
