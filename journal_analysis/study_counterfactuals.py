"""Study 1: counterfactual recourse for credit-default prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess import load_german_credit, load_give_me_credit, load_taiwan_credit


RANDOM_STATE = 42


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Any
    immutable: tuple[str, ...]
    actionable: tuple[str, ...]
    nondecreasing: tuple[str, ...] = ()
    nonincreasing: tuple[str, ...] = ()


SPECS = (
    DatasetSpec(
        "german_credit",
        load_german_credit,
        ("age_years", "personal_status_sex", "foreign_worker"),
        (
            "duration_months",
            "credit_amount",
            "savings_account_bonds",
            "installment_rate_percent",
            "other_debtors_guarantors",
            "other_installment_plans",
            "housing",
            "existing_credits_at_bank",
        ),
        nonincreasing=("duration_months", "credit_amount", "installment_rate_percent"),
    ),
    DatasetSpec(
        "give_me_some_credit",
        load_give_me_credit,
        ("age",),
        (
            "RevolvingUtilizationOfUnsecuredLines",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        ),
        nondecreasing=("MonthlyIncome",),
        nonincreasing=(
            "RevolvingUtilizationOfUnsecuredLines",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse",
        ),
    ),
    DatasetSpec(
        "taiwan_credit",
        load_taiwan_credit,
        ("SEX", "EDUCATION", "MARRIAGE", "AGE"),
        (
            "LIMIT_BAL",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ),
        nondecreasing=(
            "LIMIT_BAL",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ),
        nonincreasing=("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
    ),
)


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def build_model(X: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    transform = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = Pipeline(
        [
            ("transform", transform),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1500,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return model, numeric, categorical


def metrics(y: pd.Series, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, probability),
        "average_precision": average_precision_score(y, probability),
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "f1": f1_score(y, prediction),
        "brier_score": brier_score_loss(y, probability),
        "confusion_matrix": confusion_matrix(y, prediction).tolist(),
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "predicted_positive_rate": float(prediction.mean()),
    }


def permitted_change(spec: DatasetSpec, feature: str, old: Any, new: Any) -> bool:
    if feature not in spec.actionable or pd.isna(new):
        return False
    if feature in spec.nondecreasing and new < old:
        return False
    if feature in spec.nonincreasing and new > old:
        return False
    return old != new


def mixed_distance(
    query: pd.Series,
    candidates: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    scales: pd.Series,
) -> np.ndarray:
    distance = np.zeros(len(candidates), dtype=float)
    used = 0
    for feature in numeric:
        scale = float(scales.get(feature, 1.0))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        distance += (
            pd.to_numeric(candidates[feature], errors="coerce").fillna(query[feature])
            - float(query[feature])
        ).abs().to_numpy() / scale
        used += 1
    for feature in categorical:
        distance += (candidates[feature].astype(str) != str(query[feature])).to_numpy()
        used += 1
    return distance / max(used, 1)


def counterfactuals(
    model: Pipeline,
    spec: DatasetSpec,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    count: int,
) -> pd.DataFrame:
    test_probability = model.predict_proba(X_test)[:, 1]
    denied = X_test.loc[test_probability >= 0.5].copy()
    denied["_probability"] = test_probability[test_probability >= 0.5]
    denied = denied.sort_values("_probability", ascending=False).head(count)

    train_probability = model.predict_proba(X_train)[:, 1]
    desirable = X_train.loc[train_probability < 0.5].copy()
    # Bound nearest-neighbor cost while keeping a deterministic, diverse pool.
    if len(desirable) > 5000:
        desirable = desirable.sample(5000, random_state=RANDOM_STATE)
    scales = X_train[numeric].quantile(0.75) - X_train[numeric].quantile(0.25)
    rows: list[dict[str, Any]] = []

    for source_index, denied_row in denied.iterrows():
        source = denied_row.drop("_probability")
        distances = mixed_distance(source, desirable, numeric, categorical, scales)
        # Twenty close observed cases provide a useful plausibility/speed tradeoff.
        # Larger pools materially slow individual recourse on the 150k-row dataset.
        neighbor_order = np.argsort(distances)[:20]
        best: dict[str, Any] | None = None

        for neighbor_position in neighbor_order:
            neighbor = desirable.iloc[int(neighbor_position)]
            proposed = source.copy()
            changes = []
            options = []
            for feature in spec.actionable:
                if feature not in source or not permitted_change(
                    spec, feature, source[feature], neighbor[feature]
                ):
                    continue
                trial = proposed.copy()
                trial[feature] = neighbor[feature]
                trial_probability = model.predict_proba(
                    pd.DataFrame([trial], columns=X_train.columns)
                )[0, 1]
                gain = model.predict_proba(
                    pd.DataFrame([proposed], columns=X_train.columns)
                )[0, 1] - trial_probability
                options.append((gain, feature, neighbor[feature]))

            for _, feature, value in sorted(options, reverse=True):
                old = proposed[feature]
                proposed[feature] = value
                new_probability = model.predict_proba(
                    pd.DataFrame([proposed], columns=X_train.columns)
                )[0, 1]
                changes.append(
                    {
                        "feature": feature,
                        "from": json_value(old),
                        "to": json_value(value),
                    }
                )
                if new_probability < 0.5:
                    candidate = {
                        "source_index": int(source_index),
                        "original_default_probability": float(
                            denied_row["_probability"]
                        ),
                        "counterfactual_default_probability": float(new_probability),
                        "flipped": True,
                        "number_of_changes": len(changes),
                        "changes": json.dumps(changes),
                        "neighbor_distance": float(distances[neighbor_position]),
                    }
                    if best is None or (
                        candidate["number_of_changes"],
                        candidate["neighbor_distance"],
                    ) < (best["number_of_changes"], best["neighbor_distance"]):
                        best = candidate
                    break
        if best is None:
            best = {
                "source_index": int(source_index),
                "original_default_probability": float(denied_row["_probability"]),
                "counterfactual_default_probability": np.nan,
                "flipped": False,
                "number_of_changes": 0,
                "changes": "[]",
                "neighbor_distance": np.nan,
            }
        rows.append(best)
    return pd.DataFrame(rows)


def run_dataset(spec: DatasetSpec, output_root: Path, cf_count: int) -> None:
    print(f"\n=== {spec.name} ===", flush=True)
    frame = spec.loader()
    y = frame.pop("default").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        frame,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    model, numeric, categorical = build_model(X_train)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]

    result_dir = output_root / spec.name
    result_dir.mkdir(parents=True, exist_ok=True)
    result_metrics = metrics(y_test, probability)
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result_metrics, handle, indent=2)

    sample_size = min(5000, len(X_test))
    sampled = X_test.sample(sample_size, random_state=RANDOM_STATE)
    importance = permutation_importance(
        model,
        sampled,
        y_test.loc[sampled.index],
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).to_csv(
        result_dir / "permutation_importance.csv", index=False
    )

    cf = counterfactuals(
        model, spec, X_train, X_test, numeric, categorical, cf_count
    )
    cf.to_csv(result_dir / "counterfactuals.csv", index=False)
    with (result_dir / "dataset_spec.json").open("w", encoding="utf-8") as handle:
        payload = asdict(spec)
        payload.pop("loader")
        json.dump(payload, handle, indent=2)
    print(
        f"AUC={result_metrics['roc_auc']:.3f}; "
        f"counterfactual success={cf['flipped'].mean():.1%}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/counterfactual_study"),
        help="Output directory",
    )
    parser.add_argument(
        "--counterfactuals",
        type=int,
        default=20,
        help="High-risk test cases to explain per dataset",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=[spec.name for spec in SPECS],
        help="Optional subset; default is all datasets",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = [s for s in SPECS if not args.datasets or s.name in args.datasets]
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset_spec in selected:
        run_dataset(dataset_spec, args.output, args.counterfactuals)
