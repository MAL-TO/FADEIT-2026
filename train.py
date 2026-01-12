# train.py
import argparse
import json
from pathlib import Path
from datetime import datetime

from train_pipeline import (
    TrainConfig,
    run_training,
    run_competition_training_and_predict,
)

# -----------------------------
# OPTIONS
# -----------------------------

MODEL_CHOICES = {
    "xlm-roberta": "xlm-roberta-base",
    "umberto": "Musixmatch/umberto-commoncrawl-cased-v1",
    "alberto": "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0",
}

# Internal evaluation datasets (train/val/test)
DATASET_CHOICES = {
    "original_data": {
        "train": "splits/train_set.json",
        "val":   "splits/val_set.json",
        "test":  "splits/test_set.json",
    },
    "augmented_data": {
        "train": "splits/train_set_augmented.json",
        "val":   "splits/val_set.json",
        "test":  "splits/test_set.json",
    },
}

# Competition datasets (full training only)
COMPETITION_DATASET_CHOICES = {
    "original_full": {
        "train_full": "full_data.json",
    },
    "augmented_full": {
        "train_full": "full_data_augmented.json",
    },
}

LOSS_CHOICES = ["bce", "weighted_bce", "focal"]


# -----------------------------
# Helpers
# -----------------------------

def make_run_dir(mode: str, model_key: str, loss: str, dataset_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / mode / f"{dataset_name}__{model_key}__{loss}__{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, args: argparse.Namespace):
    cfg_path = run_dir / "cli_args.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train multi-label models for internal eval or competition prediction."
    )

    parser.add_argument("--mode", choices=["internal", "competition"], required=True)

    parser.add_argument("--model", choices=MODEL_CHOICES.keys(), required=True)
    parser.add_argument("--loss", choices=LOSS_CHOICES, required=True)

    # Internal-mode dataset choice
    parser.add_argument("--dataset", choices=DATASET_CHOICES.keys(), default=None)

    # Competition-mode dataset choice (full train json)
    parser.add_argument(
        "--competition_dataset",
        choices=COMPETITION_DATASET_CHOICES.keys(),
        default=None
    )

    # Common hyperparameters
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    # Competition-mode I/O
    parser.add_argument("--test_tsv", type=str, default=None,
                        help="Path to unlabeled competition test.tsv (required in competition mode).")
    parser.add_argument("--out_tsv", type=str, default="submission.tsv",
                        help="Where to write submission TSV (competition mode).")

    # Internal-mode outputs
    parser.add_argument("--out_json", type=str, default=None,
                        help="Where to write metrics JSON (internal mode). If omitted, saved under runs/.")

    args = parser.parse_args()

    # Create run directory
    if args.mode == "internal":
        dataset_name = args.dataset if args.dataset else "internal"
    else:
        dataset_name = args.competition_dataset if args.competition_dataset else "competition"

    run_dir = make_run_dir(args.mode, args.model, args.loss, dataset_name)
    save_config(run_dir, args)

    # Build config object for pipeline
    model_name = MODEL_CHOICES[args.model]

    # -------------------------
    # INTERNAL MODE
    # -------------------------
    if args.mode == "internal":
        if args.dataset is None:
            raise ValueError("--dataset is required for internal mode.")

        ds = DATASET_CHOICES[args.dataset]

        cfg = TrainConfig(
            model_name=model_name,
            loss_name=args.loss,
            dataset_name=args.dataset,
            train_path=ds["train"],
            val_path=ds["val"],
            test_path=ds["test"],
            max_len=args.max_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            threshold=args.threshold,
            seed=args.seed,
        )

        
        results = run_training(cfg)

        # Save results JSON
        out_json = Path(args.out_json) if args.out_json else (run_dir / "results.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nSaved internal results JSON to: {out_json}")
        print(f"Run directory: {run_dir}")

    # -------------------------
    # COMPETITION MODE
    # -------------------------
    else:
        if args.competition_dataset is None:
            raise ValueError("--competition_dataset is required for competition mode.")
        if args.test_tsv is None:
            raise ValueError("--test_tsv is required for competition mode.")

        cds = COMPETITION_DATASET_CHOICES[args.competition_dataset]

        cfg = TrainConfig(
            model_name=model_name,
            loss_name=args.loss,
            dataset_name=args.competition_dataset,
            train_path=cds["train_full"],
            val_path="",
            test_path="",
            max_len=args.max_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            threshold=args.threshold,
            seed=args.seed,
        )

        out_tsv = Path(args.out_tsv)
        if out_tsv.parent == Path("."):
            out_tsv = run_dir / out_tsv.name

        run_competition_training_and_predict(
            cfg=cfg,
            unlabeled_test_tsv=args.test_tsv,
            out_tsv=str(out_tsv)
        )

        print(f"\nSaved submission TSV to: {out_tsv}")
        print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
