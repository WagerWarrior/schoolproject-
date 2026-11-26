# Student Dropout Classification

Project repository: https://github.com/WagerWarrior/schoolproject-

## Run the simple Streamlit app

The app has three pages: Results, What mattered, and Try a made-up profile.
It displays saved aggregate results and uses the locked model for example
predictions. Inputs are not written to disk or a database. It runs locally;
the GitHub repository is source code, not a hosted app.

```bash
source .venv/bin/activate
streamlit run app.py
```

Open http://127.0.0.1:8501. Stop the server with Ctrl+C in its terminal.
Use the sidebar to switch pages. Model comparison uses initial validation
scores; the final test results are presented separately. The profile form uses
all 17 original inputs, the saved encoding/scaling pipeline and cutoff 0.3.
Scores are uncalibrated model scores, not established individual probabilities.
Impossible course-count combinations are blocked. Use fictional profiles only.

For a fresh clone, install requirements and download the source data as below.
Run scripts 01, 02, 03 and 04 to prepare the saved training split, then run
`python 19_build_app_model.py` to rebuild the locked model using training rows
only. This avoids rerunning the final test evaluation. Aggregate result files
are already in Git. The Results and What mattered pages work without the local
model; the profile page explains when the model is missing.

Run `python check_app.py` to check navigation, filters, a prediction against the
saved pipeline, the 0.3 cutoff and invalid-input handling. Requires the local model.
Streamlit form and test APIs follow the official documentation:
[forms](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form),
[AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).

The app is a simple first version. Cohort filtering and per-profile feature
contributions from the proposal are not implemented yet. A descriptive subgroup review and a Word report draft are now available; team
contributions, final packaging and usability/setup checks remain before submission.

## Current status: final B model evaluated

Interpretation completed in step 18 using final-model coefficients and
training-only validation permutation importance. No final settings changed.

The user locked logistic regression B, C=1, cutoff 0.3 before opening the held-out
test data. We fitted a fresh preprocessing/model pipeline on all 3,539 training
rows and evaluated once on the 885 test rows. No settings changed afterward.

| Final test metric | Value |
| --- | ---: |
| Accuracy | 0.8102 |
| Precision | 0.6835 |
| Recall | 0.7606 |
| F1 | 0.7200 |
| ROC-AUC | 0.8851 |
| Average precision | 0.8255 |
| PR-AUC (trapezoidal) | 0.8250 |

Test counts: 216 caught dropouts, 68 missed dropouts, 100 false alarms and 501
correct other outcomes. There were 284 actual dropouts and 601 other outcomes.
The model flagged 316 students. These results are for B only; our A/B comparison
remains based on validation, not a held-out comparison of A and B.

Configuration: `final_model_config.json`; results: `results/final_test_metrics.json`.
The local `models/final_logistic_b.joblib` stores the pipeline, input columns and
cutoff together. Its scores are not established to be calibrated probabilities.
The app must apply cutoff 0.3 to `predict_proba(... )[:, 1]`, rather than relying
on the classifier's default `predict()` cutoff. Model and individual test
prediction files are excluded from Git. The script refuses to overwrite an
existing evaluation. Do not tune on these test results.

The sections below document earlier experiments and their decisions at the time.

ITEC 3040 course project comparing enrollment information (A) with enrollment
plus first-semester results (B). Dropout is 1; Enrolled and Graduate are 0.

The project includes data checks, training-only exploration, encoding, scaling,
a baseline, three classification methods, tuning, the final B evaluation,
model interpretation, and a local Streamlit app. See Requirements Review.txt
for the remaining submission work and Code Guide.txt for a simple reading order.

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
python 16_compare_cutoffs.py
```

After locking the model settings, the final evaluation was run with
`python 17_evaluate_final_model.py`. Read the saved results rather than rerunning
it to guide further tuning. The application is built; the final report remains to do.

After the final artifact exists, `python 18_explain_model.py` explains weights
and original-column importance using training-only folds. It does not rerun
the test evaluation or overwrite the final model.

## What the model learned

We shuffled each original column in validation data and measured the drop in
average precision, repeating five times in each of five folds. This evaluates
the locked model setup using training-only refits. Bigger drops mean that
the model relied more on that column in this check.

| Original column | Mean average-precision drop |
| --- | ---: |
| First-semester courses passed | 0.4582 |
| First-semester courses enrolled | 0.0672 |
| First-semester credited courses | 0.0447 |
| Degree program | 0.0267 |
| Application mode | 0.0130 |
| Age at enrollment | 0.0098 |

These are metric decreases, not percentages of total importance. They do not
sum to one. The courses-passed coefficient is negative: holding other inputs
fixed, more passed courses lowers the model's dropout score. Numerical weights
are per one training standard deviation, not per original unit. Category
coefficients are conditional contributions, not standalone risk rankings.

Average grade's mean importance decrease was 0.0014 with repeat/fold spread
0.0024. Related columns can share information. Low incremental importance does
not mean grades never matter. Shuffling related counts can also create impossible
combinations (such as passed greater than enrolled), potentially exaggerating
importance. Treat this as a diagnostic, not a causal ranking or a reason to
change features after testing. All features and final settings remain unchanged.
See `results/column_importance_summary.csv` and `results/final_numeric_weights.csv`.

The preview scripts explain preprocessing; their fitted transformations must
not be reused for cross-validation. The training scripts fit new pipelines
within each training fold.

## Initial validation experiments (historical)

These bullets describe the initial experiments, before tuning and final testing.
The completed final evaluation is documented at the top of this README.

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

## Validation cutoff comparison

We compared score cutoffs 0.3, 0.4, 0.5 and 0.6 using saved out-of-fold
logistic predictions: A with C=10 and B with C=1. No retraining or test data.
These are separately tuned candidates; their differences do not isolate the
feature effect as strictly as the earlier matched-setting A/B comparisons.

| B cutoff | Mean precision | Mean recall | Mean F1 | Caught dropouts | False alarms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 0.666 | 0.779 | 0.718 | 886 | 445 |
| 0.4 | 0.735 | 0.696 | 0.714 | 791 | 288 |
| 0.5 | 0.798 | 0.627 | 0.702 | 713 | 181 |
| 0.6 | 0.845 | 0.565 | 0.677 | 642 | 119 |

Lower cutoffs flag more students, catching more dropouts and creating more false
alarms. Cutoff 0.3 has the highest mean F1 in this limited grid, but the error
trade-off needed justification before selection. We subsequently locked 0.3;
see the final model section above for the completed evaluation.
ROC-AUC, average precision and PR-AUC are unchanged because scores are unchanged.
See `results/cutoff_comparison.csv` for both candidates and all counts.

## Limitations

This is retrospective classification of recorded outcomes at one Portuguese
institution. No dropout dates are available to establish future dropout after
semester one. Still enrolled does not guarantee eventual graduation. Results
describe associations, not causes. Application order has a documented source
ambiguity. Parental information can act as a socioeconomic proxy and requires
fairness review. This is an educational project, not a system for real student
admission, funding or intervention decisions.

### Guided fictional example

On **Try a made-up profile**, keep the defaults and calculate: score **0.400**, above the **0.3** cutoff. Change only **Courses passed** from 4 to 6: score **0.117**, below the cutoff. These illustrate model behavior, not calibrated probabilities or causal effects. The in-app walkthrough explains each step. Styling is in `app.css`.

## Subgroup review and report draft

Run `python 20_review_subgroup_errors.py` after steps 01–04 and 14. It uses saved
validation scores for B, C=1 and reapplies cutoff 0.3. It does not retrain or load
test predictions. The output is `results/subgroup_validation_summary.csv`.
Groups below 30 students have outcome counts/rates withheld; rates with denominators
below 20 are blank. These are reporting choices, not guarantees of reliability or privacy.
Binary audit groups retain source codes pending confirmation of descriptive labels.
Gender code 0/1 false-positive rates were 15.2%/26.9%, with recall 73.2%/82.9%.
This post-selection validation review does not establish fairness or causation.

`report/Student Dropout Report Draft.docx` covers the three methods, EDA, preparation,
validation, final testing, interpretation, subgroup review, app and limitations.
The report includes the existing EDA plot. Add actual team contributions and approved
intake/dictionary documents, and complete app/setup checks before submission.
