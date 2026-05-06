## Flatbug dataset setup

The flatbug dataset must be downloaded from Zenodo and processed into YOLO format before use.

### Step 1: Download the original flatbug dataset

Download the flatbug dataset from Zenodo:

- URL: https://zenodo.org/records/14761447
- DOI: 10.5281/zenodo.14761447

Extract the dataset in this same folder (`datasets(others)/flatbug/`).

Expected location after extraction:
- `datasets(others)/flatbug/` (contains the extracted flatbug data)

### Step 2: Convert to YOLO format

Run the conversion script to transform the flatbug dataset into YOLO format:

```bash
python datasets(others)/flatbug/convert_flatbug.py
```

This generates:
- `datasets(others)/flatbug-yolo/` - YOLO-formatted flatbug dataset

### Step 3: Split validation set

If you need to further split the validation set into separate validation and test subsets:

```bash
python datasets(others)/flatbug/split_flatbug.py
```

This generates:
- `datasets(others)/flatbug-yolo-split/` - Split flatbug dataset with separate val/test directories

**Note:** The split step is optional. The `flatbug-yolo/` output from step 2 can be used directly for training or validation.
