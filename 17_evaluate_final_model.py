"""Fit the locked model on training rows, then evaluate the reserved test set."""
from pathlib import Path
import json
import warnings
from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             precision_recall_curve, auc, confusion_matrix)
from preprocessing import make_preprocessor

folder = Path(__file__).resolve().parent
output = folder / 'results'
report_path = output / 'final_test_metrics.json'
# Protect the completed evaluation from being accidentally overwritten.
if report_path.exists():
    raise SystemExit('Final evaluation already exists. Read it instead of rerunning or tuning on test data.')
config = json.loads((folder / 'final_model_config.json').read_text())
plan = json.loads((folder / 'feature_plan.json').read_text())
columns = plan['model_b_columns']
assert config['feature_set'] == 'B'

X_train = pd.read_csv(folder / 'data/split/X_train.csv', index_col='source_row_index')
y_train = pd.read_csv(folder / 'data/split/y_train.csv', index_col='source_row_index')['dropout']
assert X_train.index.equals(y_train.index)
warnings.simplefilter('error', ConvergenceWarning)

# Fit encoding, scaling and model weights using ALL training students only.
model = Pipeline([
    ('prepare', make_preprocessor('B')),
    ('classifier', LogisticRegression(C=config['C'], max_iter=config['max_iter'])),
])
model.fit(X_train[columns], y_train)

# The final test group is opened only after the configuration is locked.
X_test = pd.read_csv(folder / 'data/split/X_test.csv', index_col='source_row_index')
y_test = pd.read_csv(folder / 'data/split/y_test.csv', index_col='source_row_index')['dropout']
assert X_test.index.equals(y_test.index)
assert X_train.index.is_unique and X_test.index.is_unique
assert set(X_train.index).isdisjoint(X_test.index)
assert len(X_train) == 3539 and len(X_test) == 885
manifest = pd.read_csv(folder / 'data/split/split_manifest.csv', index_col='source_row_index')
assert manifest.loc[X_train.index, 'split'].eq('train').all()
assert manifest.loc[X_test.index, 'split'].eq('test').all()

# Score unseen test rows with the trained pipeline; no learning happens here.
scores = model.predict_proba(X_test[columns])[:, 1]
# Convert scores into labels using the cutoff chosen during validation.
predicted = (scores >= config['cutoff']).astype(int)
# Count correct other outcomes, false alarms, missed dropouts, and caught dropouts.
tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
curve_precision, curve_recall, _ = precision_recall_curve(y_test, scores)
# Store the metrics and context needed to explain the final evaluation.
report = {
    'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
    'configuration': config,
    'training_students': len(X_train), 'test_students': len(X_test),
    'actual_dropouts': int(y_test.sum()), 'actual_other_outcomes': int(y_test.eq(0).sum()),
    'accuracy': accuracy_score(y_test, predicted),
    'precision': precision_score(y_test, predicted, zero_division=0),
    'recall': recall_score(y_test, predicted, zero_division=0),
    'F1': f1_score(y_test, predicted, zero_division=0),
    'ROC_AUC': roc_auc_score(y_test, scores),
    'average_precision': average_precision_score(y_test, scores),
    'PR_AUC_trapezoidal': auc(curve_recall, curve_precision),
    'true_negative': int(tn), 'false_positive': int(fp),
    'false_negative': int(fn), 'true_positive': int(tp),
    'false_positive_rate': float(fp / (fp + tn)),
    'predicted_dropout_count': int(predicted.sum()),
    'notes': ['No fitting or setting selection used test rows.',
              'This is one held-out test evaluation, not cross-validation.',
              'Scores have not been checked for calibration.',
              'Recorded outcomes; no dropout dates to establish future forecasting.'],
}
output.mkdir(exist_ok=True)
pd.DataFrame({'source_row_index': y_test.index, 'actual': y_test.to_numpy(),
              'score': scores, 'predicted': predicted}).to_csv(
    output / 'final_test_predictions.csv', index=False)
saved = folder / 'models'
saved.mkdir(exist_ok=True)
joblib.dump({'pipeline': model, 'cutoff': config['cutoff'],
             'input_columns': columns, 'config': config}, saved / 'final_logistic_b.joblib')
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
