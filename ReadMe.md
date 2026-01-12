# repository for MALTO Fallacy Classification

This repository contains code to train and evaluate transformer-based models for **multi-label classification** of fallacies.  
It supports two workflows:

1. **Internal evaluation**: train on train split, validate on val split, evaluate on test split (with two annotator label sets).
2. **Competition / submission**: train on a full labeled dataset (original or augmented) and predict labels for an **unlabeled** `test.tsv` file, producing a submission TSV.



## 1.Setup

Install dependencies:

```bash
pip install torch transformers scikit-learn tqdm iterstrat numpy
```
If you use a GPU, make sure your installed PyTorch build supports CUDA.

## 2.Repository structure (expected)

The code expects the following structure:

```text
paper_repo/
├─ train.py
├─ train_pipeline.py
├─ full_data.json
├─ full_data_augmented.json
├─ splits/
│  ├─ train_set.json
│  ├─ val_set.json
│  ├─ test_set.json
│  ├─ train_set_augmented.json
├─ test.tsv              # (Evalita provided unlabeled test file)
```

## 3.Running the code for internal datasets(For windows CLI)


This mode is used to reproduce results for the paper and inspect performance.


### Command

```powershell
python train.py --mode internal --model alberto --loss bce --dataset original_data
```

### Options

**Models**

* `xlm-roberta` → `xlm-roberta-base`
* `umberto` → `Musixmatch/umberto-commoncrawl-cased-v1`
* `alberto` → `m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0`

**Loss functions**

* `bce`
* `weighted_bce`
* `focal`

**Internal datasets**

* `original_data` → uses `splits/train_set.json`, `splits/val_set.json`, `splits/test_set.json`
* `augmented_data` → uses `splits/train_set_augmented.json`, `splits/val_set.json`, `splits/test_set.json`


### What it prints

After training, the program prints:

* Metrics on **Test set using labels_a1**
* Metrics on **Test set using labels_a2**
* Mean metrics (average of the two)
* Per-label metrics table (precision/recall/F1/support)

### Outputs

A run folder is created under:

```text
runs/internal/<dataset>__<model>__<loss>__<timestamp>/
```

This folder typically contains:

* `cli_args.json` (the command-line arguments used)
* `results.json` (global metrics and metadata)
* optional: prediction JSON files (if enabled in `train_pipeline.py`)

---

## 4.Competition / submission mode

This mode is used to train on a **full labeled dataset** and predict on an **unlabeled** `test.tsv` file.

### Command (original full training)

```powershell
python train.py --mode competition --model xlm-roberta --loss focal --competition_dataset original_full --test_tsv test.tsv --out_tsv submission.tsv
```

### Command (augmented full training)

```powershell
python train.py --mode competition --model xlm-roberta --loss focal --competition_dataset augmented_full --test_tsv test.tsv --out_tsv submission.tsv
```

### Competition datasets

Competition mode uses *full training* JSON files only:

* `original_full` → `competition_data/train_full_original.json`
* `augmented_full` → `competition_data/train_full_augmented.json`

### Output

The output file is a TSV (e.g., `submission.tsv`) with the same columns as the input `test.tsv`, but filled with predictions:

* `labels_a1` contains predicted labels as a pipe-separated string: `Label1|Label2|...`
* `labels_a2` is filled with the same predictions by default

The file is saved either at the path you provide in `--out_tsv` or inside the run folder if you provide a filename only.

---
