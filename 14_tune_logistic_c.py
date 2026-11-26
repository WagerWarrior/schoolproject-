"""Compare three C settings using our saved training-only validation folds."""
from pathlib import Path
import json
import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_curve, auc)
from preprocessing import make_preprocessor

folder = Path(__file__).resolve().parent
# Load training inputs (X), answers (y), and the saved validation group for each row.
X = pd.read_csv(folder / 'data/split/X_train.csv', index_col='source_row_index')
y = pd.read_csv(folder / 'data/split/y_train.csv', index_col='source_row_index')['dropout']
manifest = pd.read_csv(folder / 'data/split/split_manifest.csv', index_col='source_row_index')
folds = manifest.loc[X.index, 'validation_fold']
plan = json.loads((folder / 'feature_plan.json').read_text())
assert X.index.equals(y.index) and folds.between(1, 5).all()
warnings.simplefilter('error', ConvergenceWarning)

# Collect one set of measurements per model setting and validation fold.
results = []
predictions = []
for version in ['A', 'B']:
    columns = plan[f'model_{version.lower()}_columns']
    inputs = X[columns]
    for C in [0.1, 1, 10]:
        for fold in range(1, 6):
            train_rows = folds != fold
            validation_rows = folds == fold
            # Smaller C means a stronger penalty on large weights.
            # Fit a new encoder, scaler and model inside every training fold.
            model = Pipeline([
                ('prepare', make_preprocessor(version)),
                ('classifier', LogisticRegression(C=C, max_iter=1000)),
            ])
            model.fit(inputs.loc[train_rows], y.loc[train_rows])
            scores = model.predict_proba(inputs.loc[validation_rows])[:, 1]
            predicted = (scores >= 0.5).astype(int)
            actual = y.loc[validation_rows]
            curve_precision, curve_recall, _ = precision_recall_curve(actual, scores)
            tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
            results.append({
                'version': version, 'C': C, 'fold': fold,
                'accuracy': accuracy_score(actual, predicted),
                'precision': precision_score(actual, predicted, zero_division=0),
                'recall': recall_score(actual, predicted, zero_division=0),
                'F1': f1_score(actual, predicted, zero_division=0),
                'ROC_AUC': roc_auc_score(actual, scores),
                'average_precision': average_precision_score(actual, scores),
                'PR_AUC_trapezoidal': auc(curve_recall, curve_precision),
                'true_negative': tn, 'false_positive': fp,
                'false_negative': fn, 'true_positive': tp,
            })
            predictions.append(pd.DataFrame({
                'source_row_index': actual.index, 'version': version,
                'C': C, 'fold': fold, 'actual': actual.to_numpy(),
                'score': scores, 'predicted': predicted,
            }))

# Save fold results so comparisons can be reproduced and checked later.
output = folder / 'results'
output.mkdir(exist_ok=True)
results = pd.DataFrame(results)
metrics = ['accuracy', 'precision', 'recall', 'F1', 'ROC_AUC',
           'average_precision', 'PR_AUC_trapezoidal']
results.to_csv(output / 'logistic_c_folds.csv', index=False)
results.groupby(['version', 'C'])[metrics].agg(['mean', 'std']).to_csv(
    output / 'logistic_c_summary.csv')
results.groupby(['version', 'C'])[['true_negative', 'false_positive',
                                  'false_negative', 'true_positive']].sum().to_csv(
    output / 'logistic_c_confusion_counts.csv')
pd.concat(predictions, ignore_index=True).to_csv(
    output / 'logistic_c_validation_predictions.csv', index=False)
print('Mean validation scores; cutoff 0.5, only C changes:')
print(results.groupby(['version', 'C'])[metrics].mean().round(4).to_string())
print('\nF1 mean and variation across folds:')
print(results.groupby(['version', 'C'])['F1'].agg(['mean', 'std']).round(4))
print('No test data loaded. These are tuning results, not final test scores.')
print('No candidate selected or saved as a final model.')
