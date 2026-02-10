from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keyword", type=str, required=True)
    p.add_argument("--outdir", type=str, default="reports")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    files = sorted(outdir.glob(f"metrics_{args.keyword}_*.csv"))

    if not files:
        raise FileNotFoundError(f"No metrics CSV files found for keyword '{args.keyword}' in {outdir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    metrics = ["accuracy", "f1", "precision", "recall"]
    splits = ["train", "test"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = outdir / f"boxplot_{args.keyword}_{ts}.png"

    plt.figure(figsize=(10, 6))

    # Build boxplot data in a stable order: train metrics then test metrics
    data = []
    labels = []
    for split in splits:
        dsplit = df[df["split"] == split]
        for m in metrics:
            data.append(dsplit[m].dropna().values)
            labels.append(f"{split}_{m}")

    plt.boxplot(data, labels=labels, vert=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Metrics Boxplot (keyword={args.keyword})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"Saved boxplot: {out_png}")


if __name__ == "__main__":
    main()
