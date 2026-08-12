"""Generate and evaluate diverse counterfactual explanations with DiCE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import dice_ml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Allow direct execution from this study's subdirectory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from study_counterfactuals import RANDOM_STATE, SPECS, DatasetSpec, build_model, metrics

OUTCOME = "default"
MODEL_NAMES = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "adaboost",
    "xgboost",
)


def build_named_model(
    X: pd.DataFrame, y: pd.Series, model_name: str
) -> tuple[Pipeline, list[str], list[str], dict[str, Any]]:
    """Build a consistently preprocessed classifier and its fit parameters."""
    baseline, numeric, categorical = build_model(X)
    transform = baseline.named_steps["transform"]
    fit_params: dict[str, Any] = {}
    if model_name == "logistic_regression":
        return baseline, numeric, categorical, fit_params
    if model_name == "decision_tree":
        classifier = DecisionTreeClassifier(
            class_weight="balanced", min_samples_leaf=5, random_state=RANDOM_STATE
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_name == "adaboost":
        classifier = AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE
        )
        fit_params["classifier__sample_weight"] = compute_sample_weight(
            class_weight="balanced", y=y
        )
    elif model_name == "xgboost":
        negative, positive = np.bincount(y.astype(int), minlength=2)
        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=float(negative / max(positive, 1)),
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("transform", transform), ("classifier", classifier)]), numeric, categorical, fit_params


def save_dataset_figures(
    counterfactuals: pd.DataFrame,
    query_summary: pd.DataFrame,
    output: Path,
    dataset_name: str,
) -> None:
    """Create compact, publication-ready diagnostics for one dataset."""
    if counterfactuals.empty:
        return
    valid = counterfactuals[counterfactuals["valid"]].copy()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for source_index, group in counterfactuals.groupby("source_index", sort=False):
        original = float(group["original_default_probability"].iloc[0])
        x0 = int(group["query_rank"].iloc[0])
        for _, row in group.iterrows():
            color = "#2a9d8f" if row["valid"] else "#d1495b"
            ax.plot([x0 - 0.12, x0 + 0.12], [original, row["counterfactual_default_probability"]],
                    color=color, alpha=0.55, linewidth=1.2)
            ax.scatter(x0 + 0.12, row["counterfactual_default_probability"], color=color, s=22)
        ax.scatter(x0 - 0.12, original, color="#264653", s=30, zorder=3)
    ax.axhline(0.5, color="#e76f51", linestyle="--", linewidth=1.4, label="Decision threshold")
    ax.set(xlabel="High-risk query rank", ylabel="Predicted default probability",
           title=f"DiCE counterfactual probability shifts — {dataset_name.replace('_', ' ').title()}")
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "counterfactual_probability_shifts.png", dpi=220)
    plt.close(fig)

    if not valid.empty:
        counts: dict[str, int] = {}
        for encoded in valid["changed_features"]:
            for feature in json.loads(encoded):
                counts[feature] = counts.get(feature, 0) + 1
        frequency = pd.Series(counts).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(9, max(4.5, 0.38 * len(frequency))))
        frequency.plot.barh(ax=ax, color="#457b9d")
        ax.set(xlabel="Number of valid counterfactuals", ylabel="Feature",
               title=f"Most frequently changed features — {dataset_name.replace('_', ' ').title()}")
        fig.tight_layout()
        fig.savefig(output / "feature_change_frequency.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        scatter = ax.scatter(valid["proximity"], valid["number_of_changes"],
                             c=valid["counterfactual_default_probability"], cmap="viridis_r",
                             s=48, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.set(xlabel="Mixed-type proximity distance", ylabel="Number of changed features",
               title=f"Proximity–sparsity trade-off — {dataset_name.replace('_', ' ').title()}")
        fig.colorbar(scatter, ax=ax, label="Counterfactual default probability")
        fig.tight_layout()
        fig.savefig(output / "proximity_sparsity_tradeoff.png", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    coverage = float((query_summary["valid_counterfactuals"] > 0).mean())
    validity = float(counterfactuals["valid"].mean())
    axes[0].bar(["Validity", "Query coverage"], [validity, coverage], color=["#2a9d8f", "#457b9d"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Proportion")
    axes[0].set_title("Counterfactual success")
    axes[1].bar(range(1, len(query_summary) + 1), query_summary["set_diversity"], color="#e9c46a")
    axes[1].set(xlabel="High-risk query rank", ylabel="Mean pairwise distance", title="Within-set diversity")
    fig.suptitle(dataset_name.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output / "validity_coverage_diversity.png", dpi=220)
    plt.close(fig)


def save_cross_dataset_figures(root: Path, dataset_names: list[str]) -> None:
    summaries = []
    for name in dataset_names:
        path = root / name / "summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        return
    frame = pd.DataFrame(summaries)
    frame.to_csv(root / "cross_dataset_summary.csv", index=False)
    labels = frame["dataset"].str.replace("_", " ").str.title()
    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    width = 0.36
    axes[0].bar(x - width / 2, frame["validity_rate"], width, label="Validity", color="#2a9d8f")
    axes[0].bar(x + width / 2, frame["query_coverage"], width, label="Coverage", color="#457b9d")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Success")
    axes[0].legend(frameon=False)
    axes[1].bar(x, frame["mean_sparsity_changes"], color="#f4a261")
    axes[1].set_title("Mean feature changes")
    axes[2].bar(x, frame["mean_set_diversity"], color="#e9c46a")
    axes[2].set_title("Mean within-set diversity")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20, ha="right")
    fig.suptitle("DiCE diverse counterfactual explanations across datasets")
    fig.tight_layout()
    fig.savefig(root / "dice_cross_dataset_summary.png", dpi=240)
    plt.close(fig)


def save_cross_model_figures(
    root: Path, model_names: list[str], dataset_names: list[str]
) -> None:
    """Aggregate predictive and recourse results across models and datasets."""
    rows = []
    for model_name in model_names:
        for dataset_name in dataset_names:
            result_dir = root / model_name / dataset_name
            summary_path = result_dir / "summary.json"
            metrics_path = result_dir / "model_metrics.json"
            if not summary_path.exists() or not metrics_path.exists():
                continue
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            predictive = json.loads(metrics_path.read_text(encoding="utf-8"))
            row.update({f"predictive_{k}": v for k, v in predictive.items() if k != "confusion_matrix"})
            rows.append(row)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "model_dataset_comparison.csv", index=False)
    labels = {name: name.replace("_", " ").title() for name in model_names}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    measures = [
        ("predictive_roc_auc", "ROC–AUC", (0, 1)),
        ("query_coverage", "Counterfactual query coverage", (0, 1.05)),
        ("mean_sparsity_changes", "Mean number of feature changes", None),
        ("mean_proximity", "Mean proximity distance", None),
    ]
    for panel_index, (ax, (column, title, ylim)) in enumerate(zip(axes.flat, measures)):
        pivot = frame.pivot(index="dataset", columns="model", values=column)
        pivot = pivot.reindex(index=dataset_names, columns=model_names)
        pivot.columns = [labels[c] for c in pivot.columns]
        pivot.index = [d.replace("_", " ").title() for d in pivot.index]
        pivot.plot.bar(ax=ax, width=0.82)
        ax.set(title=title, xlabel="")
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=18)
        if panel_index == 0:
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.get_legend().remove()
    fig.suptitle("Predictive performance and DiCE recourse across model families")
    fig.tight_layout()
    fig.savefig(root / "model_comparison_summary.png", dpi=240)
    plt.close(fig)


def complete_for_dice(frame: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Give DiCE complete metadata while retaining the fitted model pipeline."""
    result = frame.copy()
    for feature in result:
        if feature in numeric:
            result[feature] = pd.to_numeric(result[feature], errors="coerce")
            result[feature] = result[feature].fillna(result[feature].median())
        else:
            mode = result[feature].mode(dropna=True)
            result[feature] = result[feature].fillna(
                mode.iloc[0] if not mode.empty else "missing"
            ).astype(str)
    return result


def permitted_ranges(
    spec: DatasetSpec, query: pd.Series, reference: pd.DataFrame, numeric: list[str]
) -> dict[str, list[float]]:
    ranges = {}
    for feature in spec.actionable:
        if feature not in numeric:
            continue
        low, high = float(reference[feature].min()), float(reference[feature].max())
        value = float(query[feature])
        if feature in spec.nondecreasing:
            low = max(low, value)
        if feature in spec.nonincreasing:
            high = min(high, value)
        if low <= high:
            ranges[feature] = [low, high]
    return ranges


def changed_features(query: pd.Series, cf: pd.Series, features: list[str]) -> list[str]:
    changes = []
    for feature in features:
        old, new = query[feature], cf[feature]
        if isinstance(old, (int, float, np.number)) and isinstance(new, (int, float, np.number)):
            differs = not np.isclose(float(old), float(new), equal_nan=True)
        else:
            differs = str(old) != str(new)
        if differs:
            changes.append(feature)
    return changes


def distance(
    left: pd.Series,
    right: pd.Series,
    numeric: list[str],
    categorical: list[str],
    scales: pd.Series,
) -> float:
    parts = []
    for feature in numeric:
        scale = float(scales.get(feature, 1.0))
        scale = scale if np.isfinite(scale) and scale != 0 else 1.0
        parts.append(abs(float(left[feature]) - float(right[feature])) / scale)
    parts.extend(float(str(left[f]) != str(right[f])) for f in categorical)
    return float(np.mean(parts)) if parts else 0.0


def diversity(
    cfs: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    scales: pd.Series,
) -> float:
    if len(cfs) < 2:
        return 0.0
    values = [
        distance(cfs.iloc[i], cfs.iloc[j], numeric, categorical, scales)
        for i in range(len(cfs))
        for j in range(i + 1, len(cfs))
    ]
    return float(np.mean(values))


def run_dataset(
    spec: DatasetSpec,
    root: Path,
    query_count: int,
    total_cfs: int,
    model_name: str = "logistic_regression",
    dice_method: str = "genetic",
) -> None:
    X = spec.loader()
    y = X.pop(OUTCOME).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model, numeric, categorical, fit_params = build_named_model(X_train, y_train, model_name)
    model.fit(X_train, y_train, **fit_params)
    probabilities = model.predict_proba(X_test)[:, 1]

    output = root / spec.name
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_metrics.json").write_text(
        json.dumps(metrics(y_test, probabilities), indent=2), encoding="utf-8"
    )

    dice_train = complete_for_dice(X_train, numeric)
    dice_test = complete_for_dice(X_test, numeric)
    dice_frame = dice_train.copy()
    dice_frame[OUTCOME] = y_train.to_numpy()
    data = dice_ml.Data(
        dataframe=dice_frame, continuous_features=numeric, outcome_name=OUTCOME
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    explainer = dice_ml.Dice(data, dice_model, method=dice_method)

    denied = np.flatnonzero(probabilities >= 0.5)
    denied = denied[np.argsort(probabilities[denied])[::-1]][:query_count]
    scales = dice_train[numeric].quantile(0.75) - dice_train[numeric].quantile(0.25)
    features = X.columns.tolist()
    rows: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []

    for rank, position in enumerate(denied, 1):
        source_index = X_test.index[int(position)]
        query = dice_test.loc[[source_index]]
        np.random.seed(RANDOM_STATE + rank)
        try:
            generation_args: dict[str, Any] = {
                "total_CFs": total_cfs,
                "desired_class": 0,
                "features_to_vary": list(spec.actionable),
                "permitted_range": permitted_ranges(spec, query.iloc[0], dice_train, numeric),
                "posthoc_sparsity_param": 0.1,
                "verbose": False,
            }
            if dice_method == "genetic":
                generation_args.update(
                    proximity_weight=1.0,
                    diversity_weight=1.0,
                    sparsity_weight=0.5,
                )
            elif dice_method == "random":
                generation_args.update(
                    sample_size=5000,
                    random_seed=RANDOM_STATE + rank,
                )
            result = explainer.generate_counterfactuals(query, **generation_args)
            cfs = result.cf_examples_list[0].final_cfs_df
            cfs = cfs if cfs is not None else pd.DataFrame(columns=features + [OUTCOME])
            error = ""
        except Exception as exc:
            cfs = pd.DataFrame(columns=features + [OUTCOME])
            error = f"{type(exc).__name__}: {exc}"

        valid_count = 0
        for cf_number, (_, cf) in enumerate(cfs.iterrows(), 1):
            values = cf[features]
            cf_probability = float(model.predict_proba(pd.DataFrame([values]))[0, 1])
            changes = changed_features(query.iloc[0], values, features)
            valid = cf_probability < 0.5
            valid_count += int(valid)
            rows.append(
                {
                    "source_index": source_index,
                    "query_rank": rank,
                    "counterfactual_number": cf_number,
                    "original_default_probability": float(probabilities[int(position)]),
                    "counterfactual_default_probability": cf_probability,
                    "valid": valid,
                    "number_of_changes": len(changes),
                    "changed_features": json.dumps(changes),
                    "proximity": distance(query.iloc[0], values, numeric, categorical, scales),
                    "counterfactual": json.dumps(values.to_dict(), default=str),
                }
            )
        queries.append(
            {
                "source_index": source_index,
                "query_rank": rank,
                "requested_counterfactuals": total_cfs,
                "generated_counterfactuals": len(cfs),
                "valid_counterfactuals": valid_count,
                "set_diversity": diversity(cfs[features], numeric, categorical, scales),
                "error": error,
            }
        )

    counterfactuals, query_summary = pd.DataFrame(rows), pd.DataFrame(queries)
    counterfactuals.to_csv(output / "diverse_counterfactuals.csv", index=False)
    query_summary.to_csv(output / "query_summary.csv", index=False)
    valid = counterfactuals[counterfactuals["valid"]] if len(counterfactuals) else counterfactuals
    summary = {
        "dice_method": dice_method,
        "model": model_name,
        "dataset": spec.name,
        "queries": int(len(query_summary)),
        "requested_counterfactuals_per_query": total_cfs,
        "generated_counterfactuals": int(len(counterfactuals)),
        "valid_counterfactuals": int(len(valid)),
        "validity_rate": float(counterfactuals["valid"].mean()) if len(counterfactuals) else 0.0,
        "query_coverage": float((query_summary["valid_counterfactuals"] > 0).mean()) if len(query_summary) else 0.0,
        "mean_sparsity_changes": float(valid["number_of_changes"].mean()) if len(valid) else None,
        "mean_proximity": float(valid["proximity"].mean()) if len(valid) else None,
        "mean_set_diversity": float(query_summary["set_diversity"].mean()) if len(query_summary) else None,
        "failed_queries": int((query_summary["error"] != "").sum()) if len(query_summary) else 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_dataset_figures(counterfactuals, query_summary, output, spec.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/dice_diverse_study"))
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--total-cfs", type=int, default=4)
    parser.add_argument("--datasets", nargs="*", choices=[s.name for s in SPECS])
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=["logistic_regression"]
    )
    parser.add_argument("--dice-method", choices=["genetic"], default="genetic")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = [s for s in SPECS if not args.datasets or s.name in args.datasets]
    args.output.mkdir(parents=True, exist_ok=True)
    multiple_models = len(args.models) > 1
    for model_name in args.models:
        model_root = args.output / model_name if multiple_models else args.output
        for dataset_spec in selected:
            run_dataset(
                dataset_spec,
                model_root,
                args.queries,
                args.total_cfs,
                model_name,
                args.dice_method,
            )
        save_cross_dataset_figures(model_root, [s.name for s in selected])
    if multiple_models:
        save_cross_model_figures(args.output, args.models, [s.name for s in selected])
