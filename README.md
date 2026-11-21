# Student Dropout Classification

ITEC 3040 course project comparing enrollment information (A) with enrollment
plus first-semester results (B). Dropout is 1; Enrolled and Graduate are 0.

The project currently includes data checks, training-only exploration, one-hot
encoding, scaling, a dummy baseline, logistic regression and shallow decision
trees. Random forest, tuning, final test evaluation and the app remain to do.

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

Saved full-training models and tree rules are provisional, not selected final models.
See `Learning Notes.txt` for explanations and `Project Decision Log.txt` for decisions.

## Limitations

This is retrospective classification of recorded outcomes at one Portuguese
institution. No dropout dates are available to establish future dropout after
semester one. Still enrolled does not guarantee eventual graduation. Results
describe associations, not causes. Application order has a documented source
ambiguity. Parental information can act as a socioeconomic proxy and requires
fairness review. This is an educational project, not a system for real student
admission, funding or intervention decisions.
