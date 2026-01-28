#!/usr/bin/env python
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(jsonl_path):
    steps = []
    series_by_category = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = record.get("step")
            results = record.get("results", {})
            if isinstance(results, dict) and len(results) == 1:
                results = next(iter(results.values()))
            if step is None or not isinstance(results, dict):
                continue

            steps.append(step)
            for category, score in results.items():
                series_by_category.setdefault(category, []).append(score)

    return steps, series_by_category


def plot_categories(steps, series_by_category, output_path):
    categories = [c for c in series_by_category.keys() if c != "average_score"]
    if not categories:
        return

    n = len(categories)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for idx, category in enumerate(categories):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax.plot(steps, series_by_category[category], marker="o", linewidth=1)
        ax.set_title(category)
        ax.set_xlabel("step")
        ax.set_ylabel("score")
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_average(steps, series_by_category, output_path):
    if "average_score" not in series_by_category:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, series_by_category["average_score"], marker="o", linewidth=1)
    ax.set_title("average_score")
    ax.set_xlabel("step")
    ax.set_ylabel("score")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot MMVP eval curves from jsonl")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to mmvp_eval_results.jsonl")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots")
    args = parser.parse_args()

    jsonl_path = args.jsonl_path
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(jsonl_path))
    os.makedirs(output_dir, exist_ok=True)

    steps, series_by_category = load_jsonl(jsonl_path)
    if not steps:
        raise RuntimeError("No valid records found in jsonl.")

    categories_png = os.path.join(output_dir, "mmvp_categories_curve.png")
    average_png = os.path.join(output_dir, "mmvp_average_score_curve.png")

    plot_categories(steps, series_by_category, categories_png)
    plot_average(steps, series_by_category, average_png)


if __name__ == "__main__":
    main()
