# Diverse Counterfactual Explanations for Credit-Risk Prediction

This repository contains a reproducible comparison of DiCE counterfactual
explanations across three credit-risk datasets, five machine-learning models,
and two model-agnostic counterfactual generators.

## Publication tracks

The repository now separates two publication scopes:

- [`conference_analysis/`](conference_analysis/) contains the focused genetic
  DiCE analysis of how classifier choice shapes recourse.
- [`journal_analysis/`](journal_analysis/) contains the complete genetic versus
  random-sampling analysis and the planned methodological extensions.

The original root-level code and outputs remain as the canonical complete
research artifact.

## Research question

How do predictive model choice and counterfactual-generation method affect the
validity, availability, proximity, sparsity, and diversity of algorithmic
recourse for applicants predicted to default?

## Experimental design

Models:

- class-weighted logistic regression;
- balanced decision tree;
- balanced random forest (300 trees);
- AdaBoost (200 estimators with balanced sample weights);
- XGBoost (300 estimators with class-imbalance weighting).

DiCE generators:

- genetic optimization;
- independent random sampling (5,000 constrained candidates per query).

Datasets:

- German Credit;
- Give Me Some Credit;
- Taiwan Credit Default.

Every model uses the same stratified 75/25 train-test split (`random_state=42`).
Numeric variables are median-imputed and standardized. Categorical variables
are mode-imputed and one-hot encoded. DiCE requests four counterfactuals for
each of the ten highest-risk test cases. Dataset-specific immutable,
actionable, and directional constraints are applied.

## Headline findings

Across the 15 model-dataset combinations, random sampling achieved 100% mean
query coverage, compared with 96% for genetic optimization. Random sampling
also changed fewer features on average (2.84 versus 7.10) and produced more
diverse sets (8.34 versus 1.07). Its counterfactuals were, however, more distant
from the original cases (11.71 versus 5.99). This is a central trade-off:
random sampling found sparse and varied alternatives, while genetic search
generally produced more conservative alternatives.

XGBoost achieved the strongest ROC-AUC on all three datasets (0.809 German,
0.869 Give Me Some Credit, and 0.777 Taiwan). The single decision tree produced
the sparsest genetic counterfactuals but had the weakest predictive
discrimination.

## Repository structure

```text
.
|-- data/                         Source datasets used by the study
|-- figures/                      Main comparison figures
|-- results/
|   |-- genetic/                  Five-model genetic DiCE results
|   |-- random/                   Five-model random DiCE results
|   `-- method_comparison/        Genetic-versus-random tables and figures
|-- scripts/
|   |-- run_study.py              Model training, DiCE generation, evaluation
|   `-- compare_methods.py        Cross-method aggregation and figures
|-- preprocess.py                 Dataset loaders
|-- study_counterfactuals.py      Shared specifications and model preprocessing
`-- requirements.txt
```

Each model-dataset result directory includes predictive metrics, individual
counterfactuals, query-level summaries, aggregate metrics, probability-shift
plots, feature-change frequencies, proximity-sparsity plots, and
validity-diversity plots.

## Installation

Python 3.9 or a compatible recent Python version is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
python -m pip install -r requirements.txt
```

## Reproducing the experiments

Run these commands from the repository root.

Genetic DiCE:

```bash
python scripts/run_study.py \
  --dice-method genetic \
  --models logistic_regression decision_tree random_forest adaboost xgboost \
  --output results/genetic
```

Random-sampling DiCE:

```bash
python scripts/run_study.py \
  --dice-method random \
  --models logistic_regression decision_tree random_forest adaboost xgboost \
  --output results/random
```

Compare the completed experiments:

```bash
python scripts/compare_methods.py \
  --genetic results/genetic \
  --random results/random \
  --output results/method_comparison
```

The genetic experiment is computationally expensive because DiCE repeatedly
scores ensemble models during optimization. Runtime depends on hardware.

## Interpretation boundary

The counterfactuals explain the fitted classifiers: they identify altered
input profiles that cross a model's 0.50 decision threshold. They do not show
that making the suggested changes would causally reduce real default risk.

Some historical variables, including delinquency and bill amounts, are useful
for probing model behavior but may not represent realistic applicant actions.
Give Me Some Credit also contains special delinquency codes such as 96 and 98;
these can inflate distance-based measures when treated as ordinary counts.
Results therefore require domain, feasibility, causal, and fairness review
before any operational use.

## Data provenance

The included files are local copies of the benchmark datasets used to generate
the reported outputs. Users publishing or redistributing the repository should
verify and comply with the original providers' current terms and citation
requirements.

## Reference

Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning
classifiers through diverse counterfactual explanations. *Proceedings of the
2020 Conference on Fairness, Accountability, and Transparency*, 607-617.
https://doi.org/10.1145/3351095.3372850
