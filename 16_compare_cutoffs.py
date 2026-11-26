"""Compare cutoffs using saved out-of-fold logistic scores, not test data."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

folder = Path(__file__).resolve().parent
saved = pd.read_csv(folder / 'results/logistic_c_validation_predictions.csv')
# Current logistic candidates from the small C comparison.
settings = {'A': 10, 'B': 1}
results = []
# Compare A and B separately using their saved validation scores.
for version, C in settings.items():
    rows = saved[(saved['version'] == version) & (saved['C'] == C)]
    for cutoff in [0.3, 0.4, 0.5, 0.6]:
        for fold in range(1, 6):
            part = rows[rows['fold'] == fold]
            actual = part['actual']
            # Same scores; only the rule turning scores into labels changes.
            predicted = (part['score'] >= cutoff).astype(int)
            tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
            results.append({
                'version': version, 'C': C, 'cutoff': cutoff, 'fold': fold,
                'accuracy': accuracy_score(actual, predicted),
                'precision': precision_score(actual, predicted, zero_division=0),
                'recall': recall_score(actual, predicted, zero_division=0),
                'F1': f1_score(actual, predicted, zero_division=0),
                'true_negative': tn, 'false_alarm': fp,
                'missed_dropout': fn, 'caught_dropout': tp,
            })

results = pd.DataFrame(results)
output = folder / 'results'
results.to_csv(output / 'cutoff_folds.csv', index=False)
keys = ['version', 'C', 'cutoff']
metrics = ['accuracy', 'precision', 'recall', 'F1']
results.groupby(keys)[metrics].agg(['mean', 'std']).to_csv(output / 'cutoff_summary.csv')
counts = ['true_negative', 'false_alarm', 'missed_dropout', 'caught_dropout']
# Average fold metrics, but add student counts across folds.
comparison = results.groupby(keys)[metrics].mean().join(results.groupby(keys)[counts].sum())
comparison.to_csv(output / 'cutoff_comparison.csv')
print('Mean five-fold validation metrics and combined validation counts:')
print(comparison.round(4).to_string())
print('No retraining. No test data loaded. No final cutoff selected.')
print('ROC-AUC and precision-recall area summaries do not change: scores are unchanged.')
print('These are exploratory tuning results, not unbiased final performance estimates.')
