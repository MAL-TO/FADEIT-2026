import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import csv
from pathlib import Path

# -----------------------
# DATASETS
# -----------------------

class TrainDataset(Dataset):
    """Union of labels_a1 and labels_a2."""
    def __init__(self, data, tokenizer, mlb, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.mlb = mlb
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["post_text"]

        labels_a1 = item.get("labels_a1", [])
        labels_a2 = item.get("labels_a2", [])
        combined_labels = list(set(labels_a1) | set(labels_a2))

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        label_vec = self.mlb.transform([combined_labels])[0]
        enc["labels"] = torch.tensor(label_vec, dtype=torch.float)
        return enc


class TestDataset(Dataset):
    """Use labels from annotator 1 OR 2."""
    def __init__(self, data, tokenizer, mlb, max_len=256, labels_anot="first"):
        self.data = data
        self.tokenizer = tokenizer
        self.mlb = mlb
        self.max_len = max_len
        self.labels_anot = labels_anot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["post_text"]

        if self.labels_anot == "first":
            labels = item.get("labels_a1", [])
        elif self.labels_anot == "second":
            labels = item.get("labels_a2", [])
        else:
            raise ValueError("labels_anot must be 'first' or 'second'")

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        label_vec = self.mlb.transform([labels])[0]
        enc["labels"] = torch.tensor(label_vec, dtype=torch.float)
        return enc


# -----------------------
# MODEL
# -----------------------

class HFMeanPoolClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state.mean(dim=1)
        x = self.layer_norm(self.dropout(x))
        return self.out(x)


# -----------------------
# LOSSES
# -----------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        mod = (1 - pt).pow(self.gamma)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * mod * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_pos_weight(train_data: List[Dict[str, Any]], mlb: MultiLabelBinarizer) -> torch.Tensor:
    Y = []
    for item in train_data:
        labels_a1 = item.get("labels_a1", [])
        labels_a2 = item.get("labels_a2", [])
        combined = list(set(labels_a1) | set(labels_a2))
        Y.append(mlb.transform([combined])[0])
    Y = np.array(Y)

    pos_counts = Y.sum(axis=0)
    neg_counts = Y.shape[0] - pos_counts
    eps = 1e-8
    pos_weight = neg_counts / (pos_counts + eps)
    return torch.tensor(pos_weight, dtype=torch.float)


def make_criterion(loss_name: str, pos_weight: Optional[torch.Tensor], device: str):
    loss_name = loss_name.lower()
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_name == "weighted_bce":
        if pos_weight is None:
            raise ValueError("pos_weight is required for weighted_bce")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    if loss_name == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)
    raise ValueError("loss_name must be one of: bce, weighted_bce, focal")


# -----------------------
# METRICS
# -----------------------

def eval_metrics(model, loader, device: str, threshold=0.5):
    model.eval()
    probs_all, y_all = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            probs_all.append(probs)
            y_all.append(y)

    probs_all = np.vstack(probs_all)
    y_all = np.vstack(y_all)
    y_pred = (probs_all > threshold).astype(int)

    return {
        "macro_precision": precision_score(y_all, y_pred, average="macro", zero_division=0),
        "macro_recall":    recall_score(y_all, y_pred, average="macro", zero_division=0),
        "macro_f1":        f1_score(y_all, y_pred, average="macro", zero_division=0),
        "micro_precision": precision_score(y_all, y_pred, average="micro", zero_division=0),
        "micro_recall":    recall_score(y_all, y_pred, average="micro", zero_division=0),
        "micro_f1":        f1_score(y_all, y_pred, average="micro", zero_division=0),
    }


def mean_metrics(m1, m2):
    return {k: (m1[k] + m2[k]) / 2.0 for k in m1.keys()}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def per_label_metrics(model, loader, mlb, device, threshold=0.5):
    model.eval()
    probs_all, y_all = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            probs_all.append(probs)
            y_all.append(y)

    probs_all = np.vstack(probs_all)
    y_all = np.vstack(y_all)
    y_pred = (probs_all > threshold).astype(int)

    p, r, f1, support = precision_recall_fscore_support(
        y_all, y_pred, average=None, zero_division=0
    )

    rows = []
    for i, label in enumerate(mlb.classes_):
        rows.append({
            "label": label,
            "precision": p[i],
            "recall": r[i],
            "f1": f1[i],
            "support": int(support[i]),
        })

    return rows

def print_global_metrics(title, m):
    print(f"\n{title}")
    print(
        f"  Micro P/R/F1: "
        f"{m['micro_precision']:.3f} / {m['micro_recall']:.3f} / {m['micro_f1']:.3f}"
    )
    print(
        f"  Macro P/R/F1: "
        f"{m['macro_precision']:.3f} / {m['macro_recall']:.3f} / {m['macro_f1']:.3f}"
    )


def print_per_label_table(rows, max_labels=30):
    print("\n-------------------- Per-label metrics --------------------")
    print(f"{'Label':30s} {'P':>6s} {'R':>6s} {'F1':>6s} {'Supp':>6s}")
    print("-" * 65)

    for r in rows[:max_labels]:
        print(
            f"{r['label'][:30]:30s} "
            f"{r['precision']:6.2f} "
            f"{r['recall']:6.2f} "
            f"{r['f1']:6.2f} "
            f"{r['support']:6d}"
        )


# -----------------------
# MAIN TRAIN FUNCTION
# -----------------------

@dataclass
class TrainConfig:
    model_name: str
    loss_name: str
    dataset_name: str

    train_path: str
    val_path: str
    test_path: str

    max_len: int = 256
    batch_size: int = 16
    epochs: int = 10
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    threshold: float = 0.5
    seed: int = 42


def run_training(cfg: TrainConfig) -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = load_json(cfg.train_path)
    val_data   = load_json(cfg.val_path)
    test_data  = load_json(cfg.test_path)

    # IMPORTANT: fit mlb on union of labels from BOTH annotators across splits
    all_labels = []
    for item in (train_data + val_data + test_data):
        all_labels.append(item.get("labels_a1", []))
        all_labels.append(item.get("labels_a2", []))

    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = TrainDataset(train_data, tokenizer, mlb, max_len=cfg.max_len)
    val_ds   = TrainDataset(val_data, tokenizer, mlb, max_len=cfg.max_len)
    test1_ds = TestDataset(test_data, tokenizer, mlb, max_len=cfg.max_len, labels_anot="first")
    test2_ds = TestDataset(test_data, tokenizer, mlb, max_len=cfg.max_len, labels_anot="second")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size)
    test1_loader = DataLoader(test1_ds, batch_size=cfg.batch_size)
    test2_loader = DataLoader(test2_ds, batch_size=cfg.batch_size)

    model = HFMeanPoolClassifier(cfg.model_name, num_labels=len(mlb.classes_)).to(device)

    pos_weight = None
    if cfg.loss_name.lower() == "weighted_bce":
        pos_weight = compute_pos_weight(train_data, mlb)

    criterion = make_criterion(cfg.loss_name, pos_weight, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    best_val = -1.0
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        val_m = eval_metrics(model, val_loader, device=device, threshold=cfg.threshold)
        if val_m["micro_f1"] > best_val:
            best_val = val_m["micro_f1"]
            best_epoch = epoch

        print(f"Train loss: {total_loss/len(train_loader):.4f} | Val micro-F1: {val_m['micro_f1']:.4f}")

    
    

    test1 = eval_metrics(model, test1_loader, device, threshold=cfg.threshold)
    test2 = eval_metrics(model, test2_loader, device, threshold=cfg.threshold)
    avg  = mean_metrics(test1, test2)

    print("\n==================== TEST RESULTS ====================")
    print_global_metrics("Set 1 (Annotator A1):", test1)
    print_global_metrics("Set 2 (Annotator A2):", test2)
    print_global_metrics("Mean (A1 & A2):", avg)

    per_label = per_label_metrics(
        model, test1_loader, mlb, device, threshold=cfg.threshold
    )
    print_per_label_table(per_label)
    print("=====================================================")

    
    return {
        "config": cfg.__dict__,
        "best_val_micro_f1": best_val,
        "best_epoch": best_epoch,
        "test_set_1": test1,
        "test_set_2": test2,
        "test_avg": avg,
        "labels": list(mlb.classes_),
    }

# -----------------------
# FUNCTIONS FOR UNLABELED PREDICTIONS
# -----------------------

def _read_unlabeled_test_tsv(tsv_path: str, delimiter: str = "\t"):
    rows = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append(row)
    return rows


def _write_submission_tsv(rows, out_path: str, delimiter: str = "\t"):
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def predict_labels_for_texts(model, tokenizer, mlb, texts, device, max_len=256, batch_size=32, threshold=0.5):
    """
    Returns list[str] where each str is 'Label1|Label2|...' (possibly empty string).
    """
    model.eval()
    out_strings = []

    # simple batching
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            probs = torch.sigmoid(logits).cpu().numpy()

        preds = (probs > threshold).astype(int)
        pred_labels = mlb.inverse_transform(preds)

        for labels in pred_labels:
            # competition format expects pipe-separated labels
            out_strings.append("|".join(labels))

    return out_strings

def run_competition_training_and_predict(cfg: TrainConfig, unlabeled_test_tsv: str, out_tsv: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # IMPORTANT: cfg.train_path is now your full training json
    full_train_data = load_json(cfg.train_path)

    # Fit mlb on BOTH annotators (if present)
    all_labels = []
    for item in full_train_data:
        all_labels.append(item.get("labels_a1", []))
        all_labels.append(item.get("labels_a2", []))

    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    full_ds = TrainDataset(full_train_data, tokenizer, mlb, max_len=cfg.max_len)
    full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=True)

    model = HFMeanPoolClassifier(cfg.model_name, num_labels=len(mlb.classes_)).to(device)

    pos_weight = None
    if cfg.loss_name.lower() == "weighted_bce":
        pos_weight = compute_pos_weight(full_train_data, mlb)

    criterion = make_criterion(cfg.loss_name, pos_weight, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(full_loader) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    # Train only
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(full_loader, desc=f"[Competition] Epoch {epoch}/{cfg.epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"[Competition] Epoch {epoch}/{cfg.epochs} | Train loss: {total_loss/len(full_loader):.4f}")

    # Predict and write TSV (same as before)
    rows = _read_unlabeled_test_tsv(unlabeled_test_tsv, delimiter="\t")
    texts = [r["post_text"] for r in rows]

    pred_strings = predict_labels_for_texts(
        model=model,
        tokenizer=tokenizer,
        mlb=mlb,
        texts=texts,
        device=device,
        max_len=cfg.max_len,
        batch_size=cfg.batch_size,
        threshold=cfg.threshold,
    )

    for r, pred in zip(rows, pred_strings):
        r["labels_a1"] = pred
        r["labels_a2"] = pred

    _write_submission_tsv(rows, out_tsv, delimiter="\t")
