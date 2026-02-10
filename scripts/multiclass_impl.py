from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from fereshteh.deepl import SimpleNN, ClassTrainer


DROP_COLS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Unnamed: 0",
]

LABEL_COL = "Label"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/Android_Malware.csv")
    p.add_argument("--eta", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--keyword", type=str, default="hw02")
    p.add_argument("--outdir", type=str, default="reports")
    p.add_argument("--save-onnx", action="store_true")
    return p.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main():
    args = parse_args()
    device = pick_device(args.device)

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df = df.drop(columns=DROP_COLS, errors="ignore")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column '{LABEL_COL}'.")

    # X numeric only (if any non-numeric sneaks in, coerce -> NaN then fill)
    y_raw = df[LABEL_COL].astype(str)
    X = df.drop(columns=[LABEL_COL])

    # Convert everything to numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    classes = sorted(y_raw.unique().tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(class_to_id).astype(int).values

    print("Classes:", classes)
    print("m =", len(classes))

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = SimpleNN(in_features=X_train_t.shape[1], num_classes=len(classes))

    trainer = ClassTrainer(
        X_train=X_train_t,
        Y_train=y_train_t,
        eta=args.eta,
        epoch=args.epochs,
        optimizer_name=args.optimizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    print("Device:", device)
    trainer.train()

    # train metrics
    y_train_pred = trainer.predict(X_train_t)
    train_m = compute_metrics(y_train, y_train_pred)

    # test metrics
    test_m = trainer.test(X_test_t, y_test_t)

    # plots + confusion matrices
    tag = f"{args.keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trainer.evaluation(class_names=classes, outdir=outdir, tag=tag)

    if args.save_onnx:
        trainer.save(str(outdir / f"{tag}.onnx"))

    # Save CSV metrics (must include keyword + timestamp)
    rows = []
    for split, m in [("train", train_m), ("test", test_m)]:
        rows.append({
            "keyword": args.keyword,
            "timestamp": tag.split("_", 1)[1],
            "split": split,
            **m,
            "eta": args.eta,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
        })

    metrics_df = pd.DataFrame(rows)
    out_csv = outdir / f"metrics_{tag}.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"Saved metrics CSV: {out_csv}")


if __name__ == "__main__":
    main()
