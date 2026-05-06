## flatbug-yolo-split dataset requirement

Before using scripts in this folder, you must first build the YOLO dataset in the sibling flatbug folder.

Required first step:

- Build the YOLO dataset in `datasets(others)/flatbug/` by running `convert_flatbug.py`
- Split the validation set by running `split_flatbug.py`

This folder depends on that output and will not work correctly if the flatbug dataset has not been prepared first.
