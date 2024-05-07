# GSE172127 - fetal liver

## DOI
[10.1038/s41421-021-00266-1](https://doi.org/10.1038/s41421-021-00266-1)

* Whole fetal liver scRNA sequencing, and
* fetal liver HSC scRNA sequencing

## Downloading data
```
python3 download_Lu_2021.py
```

This script automatically saves the raw files in a `raw` subdirectory in `DefaultPaths.DISSOCIATED`, if you want to save the raw data somewhere else, you can provide an additional system argument.

## Processing data
```
python3 preprocess_Lu_2021.py
```

This script requires `python3 download_Lu_2021.py` to be run first. Preprocessed files are stored in a `preprocessed` subdirectory in `DefaultPaths.DISSOCIATED`, if you deposited the raw data somewhere else, please provide an additional system argument.