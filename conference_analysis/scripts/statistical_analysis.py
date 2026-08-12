"""Query-level uncertainty summaries for the genetic-DiCE conference study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


MODELS = ("logistic_regression", "decision_tree", "random_forest", "adaboost", "xgboost")
DATASETS = ("german_credit", "give_me_some_credit", "taiwan_credit")


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - radius, centre + radius


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("conference_analysis/results/genetic"))
    parser.add_argument("--output", type=Path, default=Path("conference_analysis/results/statistical_summary.csv"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []

    for model in MODELS:
        for dataset in DATASETS:
            folder = args.results / model / dataset
            queries = pd.read_csv(folder / "query_summary.csv")
            counterfactuals = pd.read_csv(folder / "diverse_counterfactuals.csv")
            valid = counterfactuals[counterfactuals["valid"] == True].copy()  # noqa: E712

            # Counterfactuals are nested within queries. Aggregate first so that
            # one prolific query cannot dominate uncertainty estimates.
            query_metrics = valid.groupby("query_rank").agg(
                mean_changes=("number_of_changes", "mean"),
                mean_proximity=("proximity", "mean"),
            )
            query_frame = queries.set_index("query_rank").join(query_metrics)
            covered = int((query_frame["valid_counterfactuals"] > 0).sum())
            coverage_low, coverage_high = wilson_interval(covered, len(query_frame))
            changes = bootstrap_mean(query_frame["mean_changes"].to_numpy(float), rng, args.bootstrap)
            proximity = bootstrap_mean(query_frame["mean_proximity"].to_numpy(float), rng, args.bootstrap)
            diversity = bootstrap_mean(query_frame["set_diversity"].to_numpy(float), rng, args.bootstrap)

            rows.append({
                "model": model,
                "dataset": dataset,
                "queries": len(query_frame),
                "covered_queries": covered,
                "coverage": covered / len(query_frame),
                "coverage_ci_low": coverage_low,
                "coverage_ci_high": coverage_high,
                "mean_changes_query_level": changes[0],
                "mean_changes_ci_low": changes[1],
                "mean_changes_ci_high": changes[2],
                "mean_proximity_query_level": proximity[0],
                "mean_proximity_ci_low": proximity[1],
                "mean_proximity_ci_high": proximity[2],
                "mean_diversity_query_level": diversity[0],
                "mean_diversity_ci_low": diversity[1],
                "mean_diversity_ci_high": diversity[2],
                "bootstrap_repetitions": args.bootstrap,
                "bootstrap_seed": args.seed,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    metadata = {
        "unit_of_analysis": "query applicant",
        "coverage_interval": "two-sided 95% Wilson score interval",
        "continuous_intervals": "query-level nonparametric percentile bootstrap",
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_seed": args.seed,
        "interpretation": "exploratory; model-specific query cohorts are not paired across classifiers",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
