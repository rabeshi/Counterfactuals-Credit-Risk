"""Compare completed DiCE genetic and random-sampling experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "model_dataset_comparison.csv")
    frame["dice_method"] = method
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genetic", type=Path, default=Path("results/genetic"))
    parser.add_argument("--random", type=Path, default=Path("results/random"))
    parser.add_argument("--output", type=Path, default=Path("results/method_comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = pd.concat(
        [load_results(args.genetic, "genetic"), load_results(args.random, "random")],
        ignore_index=True,
    )
    frame.to_csv(args.output / "genetic_vs_random.csv", index=False)

    models = ["logistic_regression", "decision_tree", "random_forest", "adaboost", "xgboost"]
    datasets = ["german_credit", "give_me_some_credit", "taiwan_credit"]
    methods = ["genetic", "random"]
    measures = [
        ("query_coverage", "Query coverage"),
        ("mean_sparsity_changes", "Mean feature changes"),
        ("mean_proximity", "Mean proximity distance"),
        ("mean_set_diversity", "Mean within-set diversity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = {"genetic": "#457b9d", "random": "#e76f51"}
    width = 0.36
    x = np.arange(len(models))
    for panel, (ax, (measure, title)) in enumerate(zip(axes.flat, measures)):
        # Average across datasets; dataset-level values remain in the CSV.
        aggregate = frame.groupby(["model", "dice_method"])[measure].mean()
        for offset, method in enumerate(methods):
            values = [aggregate.get((model, method), np.nan) for model in models]
            ax.bar(x + (offset - 0.5) * width, values, width, label=method.title(), color=colors[method])
        ax.set_title(title)
        ax.set_xticks(x, [m.replace("_", " ").title() for m in models], rotation=18, ha="right")
        if panel == 0:
            ax.legend(frameon=False)
    fig.suptitle("DiCE genetic optimization versus random sampling\n(mean across three credit datasets)")
    fig.tight_layout()
    fig.savefig(args.output / "genetic_vs_random_summary.png", dpi=240)
    plt.close(fig)

    # Dataset-stratified coverage makes failures visible without averaging them away.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, dataset in zip(axes, datasets):
        subset = frame[frame["dataset"] == dataset]
        for offset, method in enumerate(methods):
            indexed = subset[subset["dice_method"] == method].set_index("model")
            values = [indexed["query_coverage"].get(model, np.nan) for model in models]
            ax.bar(x + (offset - 0.5) * width, values, width, label=method.title(), color=colors[method])
        ax.set_title(dataset.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x, [m.replace("_", " ").title() for m in models], rotation=25, ha="right")
    axes[0].set_ylabel("Query coverage")
    axes[0].legend(frameon=False)
    fig.suptitle("Counterfactual coverage by DiCE method, classifier, and dataset")
    fig.tight_layout()
    fig.savefig(args.output / "genetic_vs_random_coverage.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
