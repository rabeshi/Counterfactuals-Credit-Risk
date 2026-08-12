# Conference Analysis: Genetic DiCE

## Proposed title

**Model Choice and Counterfactual Recourse in Credit-Risk Prediction: A
Multi-Dataset Evaluation Using DiCE**

## Scope

This folder contains the focused conference analysis. It evaluates only DiCE
genetic optimization and asks:

> How does predictive model choice affect the availability and characteristics
> of counterfactual recourse in credit-risk classification when the explanation
> generator is held constant?

The experiment covers three credit datasets and five classifiers:

- class-weighted logistic regression;
- balanced decision tree;
- balanced random forest;
- AdaBoost with balanced sample weights;
- XGBoost with class-imbalance weighting.

The saved analysis includes validity, query coverage, sparsity, proximity,
diversity, predictive performance, individual counterfactuals, and visual
diagnostics. Random-sampling results are intentionally excluded from this
publication track.

## Contents

```text
conference_analysis/
|-- figures/                 Main conference comparison figure
|-- results/genetic/         Complete genetic DiCE outputs
|-- scripts/run_study.py     Reproducible experiment
|-- preprocess.py            Dataset loading
|-- study_counterfactuals.py Shared constraints and preprocessing
`-- requirements.txt
```

The source datasets remain in the repository-level `data/` directory to avoid
duplicating large files.

## Reproduce

Run from the repository root:

```bash
python conference_analysis/scripts/run_study.py \
  --dice-method genetic \
  --models logistic_regression decision_tree random_forest adaboost xgboost \
  --output conference_analysis/results/genetic
```

## Conference-paper plan

The main manuscript should contain:

1. The credit-recourse problem and model-dependence hypothesis.
2. One dataset and model table.
3. One predictive-performance table.
4. One genetic-counterfactual performance table.
5. The cross-model comparison figure.
6. A small number of applicant-level examples.
7. Feasibility, causal, fairness, and data-quality limitations.

Detailed model-dataset plots and individual records belong in supplementary
material or this repository.

## Required strengthening before submission

The current results use each model's ten highest-risk test cases. A stronger
conference experiment should use a larger shared applicant cohort across all
models, repeat stochastic generation across seeds, report uncertainty, and
clean the special 96/98 delinquency codes in Give Me Some Credit. Raw proximity
and diversity values should be reported by dataset or normalized before
cross-dataset aggregation.

## Interpretation boundary

These counterfactuals describe changes that cross a classifier's decision
boundary. They do not establish that the changes would causally lower default
risk or be operationally available to an applicant.
