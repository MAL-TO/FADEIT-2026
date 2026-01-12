"""
The preprocessing code provided here is just to show how the data was splitted.
There is no need to run it for the model training and evaluation.
"""


import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def tsv_to_json(
    tsv_path: str,
    json_path: str,
    *,
    delimiter: str = "\t",
    label_sep: str = "|",
    encoding: str = "utf-8",
) -> List[Dict[str, Any]]:
    
    tsv_path = str(tsv_path)
    json_path = str(json_path)

    data: List[Dict[str, Any]] = []

    def _split_labels(cell: Optional[str]) -> List[str]:
        if cell is None:
            return []
        cell = cell.strip()
        if not cell:
            return []
        return [lbl.strip() for lbl in cell.split(label_sep) if lbl.strip()]

    with open(tsv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        required = {"post_id", "post_date", "post_topic_keywords", "post_text", "labels_a1", "labels_a2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required TSV columns: {sorted(missing)}. Found: {reader.fieldnames}")

        for row in reader:
            item = {
                "post_id": row["post_id"],
                "post_date": row["post_date"],
                "post_topic_keywords": row["post_topic_keywords"],
                "post_text": row["post_text"],
                "labels_a1": _split_labels(row.get("labels_a1")),
                "labels_a2": _split_labels(row.get("labels_a2")),
            }
            data.append(item)

    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def stratified_json_split(
    json_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    *,
    test_size: float = 0.30,   # first split: train vs temp
    val_size: float = 0.50,    # second split: temp -> val vs test (half/half)
    seed: int = 42,
    encoding: str = "utf-8",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    
    with open(json_path, "r", encoding=encoding) as f:
        data: List[Dict[str, Any]] = json.load(f)

    if not data:
        raise ValueError("JSON dataset is empty.")

    labels_a1 = [item.get("labels_a1", []) for item in data]
    labels_a2 = [item.get("labels_a2", []) for item in data]

    # Use union of labels across BOTH annotator sets (safer than only labels_a1)
    unique_labels = sorted(set(l for labels in (labels_a1 + labels_a2) for l in labels))

    def encode_multi_hot(label_list: List[str], all_labels: List[str]) -> List[int]:
        label_set = set(label_list)
        return [1 if lbl in label_set else 0 for lbl in all_labels]

    Y1 = np.array([encode_multi_hot(l, unique_labels) for l in labels_a1], dtype=np.int32)
    Y2 = np.array([encode_multi_hot(l, unique_labels) for l in labels_a2], dtype=np.int32)

    # Combine both label sets for stratification
    Y_all = np.concatenate([Y1, Y2], axis=1)
    X = np.arange(len(data))  # dummy indices

    # Split: train vs temp
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, temp_idx = next(msss1.split(X, Y_all))

    # Split: temp -> val vs test
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_size), random_state=seed)
    # NOTE: here test_size refers to "test portion" of temp. (1 - val_size) ensures val_size goes to val.
    val_rel_idx, test_rel_idx = next(msss2.split(X[temp_idx], Y_all[temp_idx]))

    train_data = [data[i] for i in train_idx]
    val_data = [data[temp_idx[i]] for i in val_rel_idx]
    test_data = [data[temp_idx[i]] for i in test_rel_idx]

    # Save
    for out_path, subset in [(train_path, train_data), (val_path, val_data), (test_path, test_data)]:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding=encoding) as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data



tsv_to_json("train-dev.tsv", "full_data.json")


stratified_json_split(
    "full_data.json",
    "splits/train_set.json",
    "splits/val_set.json",
    "splits/test_set.json",
    test_size=0.30,
    val_size=0.50,
    seed=42,
)
