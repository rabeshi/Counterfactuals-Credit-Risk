# Journal Analysis: Complete DiCE Comparison

## Proposed title

**How Model and Search Choices Shape Algorithmic Recourse: A Multi-Dataset
Evaluation of DiCE in Credit-Risk Prediction**

## Scope

This folder contains the complete journal analysis. It evaluates the joint
effects of predictive model choice and DiCE generation strategy:

- genetic optimization;
- independent random sampling with 5,000 constrained candidates per query.

Both generators are evaluated across German Credit, Give Me Some Credit, and
Taiwan Credit using logistic regression, decision tree, random forest,
AdaBoost, and XGBoost classifiers.

The central question is:

> How do classifier family and counterfactual-generation strategy jointly
> affect the availability, proximity, sparsity, diversity, and stability of
> credit-risk recourse?

## Contents

```text
journal_analysis/
|-- figures/                    Main model and method figures
|-- results/
|   |-- genetic/                Genetic DiCE outputs
|   |-- random/                 Random-sampling outputs
|   `-- method_comparison/      Direct method comparison
|-- scripts/
|   |-- run_study.py            Model and DiCE experiments
|   `-- compare_methods.py      Genetic-versus-random analysis
|-- preprocess.py
|-- study_counterfactuals.py
`-- requirements.txt
```

The source datasets remain in the repository-level `data/` directory.

## Reproduce

Run from the repository root.

Genetic optimization:

```bash
python journal_analysis/scripts/run_study.py \
  --dice-method genetic \
  --models logistic_regression decision_tree random_forest adaboost xgboost \
  --output journal_analysis/results/genetic
```

Random sampling:

```bash
python journal_analysis/scripts/run_study.py \
  --dice-method random \
  --models logistic_regression decision_tree random_forest adaboost xgboost \
  --output journal_analysis/results/random
```

Method comparison:

```bash
python journal_analysis/scripts/compare_methods.py \
  --genetic journal_analysis/results/genetic \
  --random journal_analysis/results/random \
  --output journal_analysis/results/method_comparison
```

## Current headline finding

Across 15 model-dataset combinations, random sampling achieved complete query
coverage and changed fewer features, but its counterfactuals were substantially
farther from the original applicants. Genetic search produced more conservative
profiles but had several generation failures. This reveals a meaningful
coverage-sparsity-proximity trade-off rather than a universally superior
generator.

## Journal extension plan

The journal study should extend the conference analysis through:

1. A larger shared query cohort across classifiers.
2. Multiple stochastic seeds and stability estimates.
3. Bootstrap confidence intervals and paired statistical comparisons.
4. Runtime and computational-efficiency analysis.
5. Random-sampling sensitivity at 1,000, 5,000, and 10,000 candidates.
6. Strict and permissive actionability policies.
7. Data cleaning and sensitivity analysis for special delinquency codes.
8. Dataset-normalized proximity and diversity comparisons.
9. Subgroup analysis of unequal access to recourse.
10. Stronger plausibility and feasibility evaluation.

## Interpretation boundary

The explanations characterize model behavior and are not causal guarantees.
Historical repayment and billing variables may be useful for model auditing
without representing realistic applicant actions.

