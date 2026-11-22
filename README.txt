STUDENT DROPOUT PROJECT — PYTHON WORKFLOW

Current stage: data checks, 80/20 split and training-only exploration. No trained model yet.
Plan follows the saved completed proposal. Fresh eClass/Drive revisions have
not been checked during this step.

Run scripts in order with the project's Python environment:
  .venv/bin/python 01_clean_unknown_codes.py
  .venv/bin/python 02_check_numeric_columns.py
  .venv/bin/python 03_check_enrollment_categories.py
  .venv/bin/python 04_split_data.py
  .venv/bin/python 05_explore_training_data.py
  .venv/bin/python 06_preview_one_hot_encoding.py

These commands assume your terminal is in this project folder.
Dependencies are recorded in requirements.txt. The raw source CSV and verified
dictionary now live in data/raw and resources, as referenced by the scripts.
See README.md for setup and source download instructions.

WHAT EACH STEP DOES
1. Preserve unknown parental categories and decode their documented labels.
2. Check numeric values and relationships, keeping valid zeros.
3. Check remaining enrollment categories and decode names.
4. Separate inputs from answers, split students, and save shared CV membership.

SPLIT FILES
data/split/X_train.csv: training inputs, all 17 Model B columns.
data/split/y_train.csv: corresponding binary answers.
data/split/X_test.csv and y_test.csv: holdout inputs and answers.
data/split/split_manifest.csv: source row index, train/test, validation fold.
Read X/y with index_col='source_row_index'. This index is bookkeeping, NOT a
model input. Model A selects the 11 columns from feature_plan.json. Model B
selects all 17. Both use identical students and validation folds.
For CV round k, fit on training rows where validation_fold != k and validate
on training rows where validation_fold == k. Test rows have fold -1 and must
never be included in CV. Fit transformations only within each training fold.

NEXT STEPS (NOT IMPLEMENTED YET)
5. DONE: Explore training data; outputs saved in analysis/.
6. Build sklearn pipelines: one-hot encoding for category labels, numerical
   scaling for logistic regression, any needed imputation fitted in each fold.
   Keep application-order documentation issue visible. Review optional derived
   pass rate from the proposal before adding it to the current 17-input plan.
7. Train dummy baseline, logistic regression, shallow tree and random forest
   using both A and B. Tune with the saved training-only five-fold split.
8. Choose models and thresholds using validation results. Lock decisions,
   refit on all training rows, then evaluate once on the reserved test rows.
   Report accuracy, precision, recall, F1, ROC-AUC, PR-AUC and confusion matrix.
9. Save fitted pipeline, create local Streamlit app, report and final README.

This is retrospective classification of recorded outcomes. No dropout dates
are available to validate a future-dropout forecast after semester one.

ONE-HOT ENCODING
preprocessing.py provides make_encoder(A or B), returning a fresh unfitted
transformer for a future model pipeline. Step 6 is a teaching preview only.
Never cross-validate a matrix already encoded using the full training group.
Fit the pipeline separately within each fold. Scaling is now included in make_preprocessor for logistic regression.

SCALING PREVIEW
Run .venv/bin/python 07_preview_scaling.py to see training-only examples.
make_preprocessor(A or B) combines encoding, standard scaling of numerical
inputs and unchanged binary fields. Use a fresh instance inside model pipelines.
Preview settings are explanatory outputs, not inputs for cross-validation.

INITIAL MODEL TRAINING
Run .venv/bin/python 08_train_first_models.py. Uses training files only.
Results are in results/; provisional fitted models are in models/.
Five-fold means and standard deviations are validation results, not final test
scores. The initial cutoff is 0.5. No parameters or thresholds tuned yet.
Average precision and trapezoidal PR-AUC are labelled separately. A constant
score baseline has a misleadingly high trapezoidal area from interpolation;
use its average precision as the prevalence-level precision-recall reference.
Next: review errors, compare tree/random forest, select using validation, then
evaluate the final locked model on the test group.

RANDOM FOREST
Run .venv/bin/python 11_train_random_forest.py. Simple initial setup: 100 trees,
seed 42, default depth, cutoff 0.5. Same A/B folds; no scaling or tuning.
Results in results/random_forest_*.csv; fitted models remain local in models/.
Current initial comparisons include all three methods. Tuning and final test
evaluation remain to do. See README.md and Learning Notes.txt for results.
