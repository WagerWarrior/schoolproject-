# Student Dropout Classification

Project repository: https://github.com/WagerWarrior/schoolproject-

ITEC 3040 course project comparing enrollment information (A) with enrollment
plus first-semester results (B). Dropout is 1; Enrolled and Graduate are 0.

The project currently includes data checks, training-only exploration, one-hot
encoding, scaling, a dummy baseline, logistic regression and shallow decision
trees and random forests. Tuning, final test evaluation and the app remain to do.

## Setup

Use Python 3.12. In the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the assigned `dataset.csv` from the
[Kaggle dataset page](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
and place it in `data/raw/dataset.csv` (create these folders if needed).
Use the assigned 35-column file with 4,424 rows. Do not substitute the UCI file:
it has different category codes and additional columns.

Data, virtual environments, row-level predictions and fitted model files are
excluded from Git. The verified category mapping in `resources/` documents
the correspondence between the assigned codes and the
[UCI source](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

## Run in order

```bash
python 01_clean_unknown_codes.py
python 02_check_numeric_columns.py
python 03_check_enrollment_categories.py
python 04_split_data.py
python 05_explore_training_data.py
python 06_preview_one_hot_encoding.py
python 07_preview_scaling.py
python 08_train_first_models.py
python 09_review_errors.py
python 10_train_decision_tree.py
python 11_train_random_forest.py
python 12_try_tree_depth5.py
python 13_try_tree_depth7.py
python 14_tune_logistic_c.py
python 15_tune_forest_leaf.py
```

The preview scripts explain preprocessing; their fitted transformations must
not be reused for cross-validation. The training scripts fit new pipelines
within each training fold.

## Evaluation

- Stratified 80/20 split, seed 42: 3,539 training students and 885 test students.
- Identical five-fold training-only validation for A and B.
- Initial prediction cutoff 0.5; no tuning or class weighting yet.
- Logistic regression uses one-hot encoding and numerical standardization.
- Trees use one-hot encoding without scaling, maximum depth 3, seed 42.
- Random forests use one-hot encoding without scaling, 100 trees, seed 42,
  default unrestricted depth, bootstrap sampling and random feature subsets.
- Results are validation results, not final test results. Test data remains reserved.
- Average precision and trapezoidal PR-AUC are recorded separately. Trapezoidal
  PR-AUC can be misleading for constant-score baselines because of interpolation.

| Initial model | Mean validation F1 | Mean validation accuracy |
| --- | ---: | ---: |
| Baseline | 0.000 | 67.9% |
| Logistic A | 0.459 | 72.3% |
| Logistic B | 0.702 | 82.9% |
| Tree A | 0.503 | 70.5% |
| Tree B | 0.642 | 80.1% |
| Forest A | 0.466 | 72.0% |
| Forest B | 0.667 | 81.2% |

Saved full-training models and tree rules are provisional, not selected final models.
See `Learning Notes.txt` for explanations and `Project Decision Log.txt` for decisions.

## First tuning comparison: depth 3 versus 5

Only tree depth changed; folds, features, seed and cutoff remained the same.
Depth-3 results are preserved. See `results/tree_depth_comparison.csv`.

| Inputs | Depth-3 mean F1 | Depth-5 mean F1 |
| --- | ---: | ---: |
| A: enrollment | 0.503 | 0.491 |
| B: plus semester one | 0.642 | 0.675 |

Depth 5 improved B in four of five folds, but not A. B caught 700 dropouts
versus 637, with 233 false alarms versus 206. Its F1 standard deviation across
folds rose from 0.025 to 0.033. Logistic B still has a higher initial mean F1
(0.702). No final model or test result is claimed.

## Depth-7 comparison

We also tried depth 7 with the same folds, seed, input sets and cutoff.

| Inputs | Depth 3 F1 | Depth 5 F1 | Depth 7 F1 |
| --- | ---: | ---: | ---: |
| A: enrollment | 0.503 | 0.491 | 0.496 |
| B: plus semester one | 0.642 | 0.675 | 0.651 |

Depth 5 remains B's best tested depth by mean validation F1; depth 3 remains
A's best. Increasing depth to 7 did not improve B. This alone does not prove
overfitting. Logistic B still leads these candidates on mean validation F1.
The test group remains reserved. Run step 13 after steps 10 and 12 to rebuild
the complete three-depth comparison.

## Logistic regression C comparison

We compared `C=0.1`, `1` and `10` with the same folds and cutoff 0.5.
Smaller C applies a stronger penalty on large weights. Encoding and scaling
were fitted separately within each training fold. Only C changed.

| Inputs | C=0.1 mean F1 | C=1 mean F1 | C=10 mean F1 |
| --- | ---: | ---: | ---: |
| A: enrollment | 0.4121 | 0.4585 | 0.4676 |
| B: plus semester one | 0.6954 | 0.7017 | 0.7014 |

For B, C=1 and C=10 are effectively tied at this precision; the tiny difference
is not evidence of a meaningful improvement. Keep C=1 as the current B
candidate. C=10 leads A on mean F1 at cutoff 0.5, but not on every metric.
These are tuning results, not unbiased final performance estimates. No final
model was selected or tested. See `results/logistic_c_summary.csv` for fold
means and standard deviations.

## Random forest leaf-size comparison

We compared random forest `min_samples_leaf=1,5,10` with 100 trees, seed 42,
the same five folds and cutoff 0.5. Larger leaves require more training samples
in each endpoint. Encoding is fitted within each fold; no scaling is used.

| Inputs | Leaf size 1 F1 | Leaf size 5 F1 | Leaf size 10 F1 |
| --- | ---: | ---: | ---: |
| A: enrollment | 0.4661 | 0.4191 | 0.3927 |
| B: plus semester one | 0.6667 | 0.6536 | 0.6494 |

Leaf size 1 retains the highest mean F1 at cutoff 0.5 for both input sets.
Larger leaves improved precision but reduced recall. They did not improve all
metrics together. These tuning results do not establish final test performance.
See `results/forest_leaf_summary.csv` for all metrics and fold variation.

## Limitations

This is retrospective classification of recorded outcomes at one Portuguese
institution. No dropout dates are available to establish future dropout after
semester one. Still enrolled does not guarantee eventual graduation. Results
describe associations, not causes. Application order has a documented source
ambiguity. Parental information can act as a socioeconomic proxy and requires
fairness review. This is an educational project, not a system for real student
admission, funding or intervention decisions.
