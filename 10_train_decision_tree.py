"""Train shallow decision trees using the same training-only validation folds."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_curve, auc)
from preprocessing import make_encoder

folder = Path(__file__).resolve().parent
X = pd.read_csv(folder / 'data/split/X_train.csv', index_col='source_row_index')
y = pd.read_csv(folder / 'data/split/y_train.csv', index_col='source_row_index')['dropout']
manifest = pd.read_csv(folder / 'data/split/split_manifest.csv', index_col='source_row_index')
folds = manifest.loc[X.index, 'validation_fold']
assert X.index.equals(y.index) and folds.between(1, 5).all()
plan = json.loads((folder / 'feature_plan.json').read_text())

# A shallow tree asks at most three questions along each path.
# Trees use cutoffs, so numerical scaling is unnecessary.
def make_model(name):
    feature_set = 'A' if name == 'Tree A' else 'B'
    return Pipeline([
        ('prepare', make_encoder(feature_set)),
        ('classifier', DecisionTreeClassifier(max_depth=3, random_state=42)),
    ])

results = []
predictions = []
names = ['Tree A', 'Tree B']
for name in names:
    columns = plan['model_a_columns'] if name == 'Tree A' else plan['model_b_columns']
    inputs = X[columns]
    for fold in range(1, 6):
        # Learn on four folds, then predict the fifth. Repeat five times.
        train_rows = folds != fold
        validation_rows = folds == fold
        model = make_model(name)
        model.fit(inputs.loc[train_rows], y.loc[train_rows])
        scores = model.predict_proba(inputs.loc[validation_rows])[:, 1]
        predicted = (scores >= 0.5).astype(int)
        actual = y.loc[validation_rows]
        curve_precision, curve_recall, _ = precision_recall_curve(actual, scores)
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        results.append({
            'model': name, 'fold': fold,
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
            'source_row_index': actual.index, 'model': name, 'fold': fold,
            'actual': actual.to_numpy(), 'score': scores, 'predicted': predicted,
        }))

output = folder / 'results'
output.mkdir(exist_ok=True)
results = pd.DataFrame(results)
results.to_csv(output / 'decision_tree_folds.csv', index=False)
metrics = ['accuracy', 'precision', 'recall', 'F1', 'ROC_AUC', 'average_precision', 'PR_AUC_trapezoidal']
summary = results.groupby('model')[metrics].agg(['mean', 'std'])
summary.to_csv(output / 'decision_tree_summary.csv')
pd.concat(predictions, ignore_index=True).to_csv(output / 'decision_tree_validation_predictions.csv', index=False)
results.groupby('model')[['true_negative', 'false_positive', 'false_negative', 'true_positive']].sum().to_csv(output / 'decision_tree_confusion_counts.csv')

# Fit initial models on all training rows for later reuse, not final test evaluation.
# These are provisional: we have not compared random forests or selected a final model.
saved = folder / 'models'
saved.mkdir(exist_ok=True)
for name in names:
    columns = plan['model_a_columns'] if name == 'Tree A' else plan['model_b_columns']
    model = make_model(name)
    model.fit(X[columns], y)
    joblib.dump(model, saved / (name.lower().replace(' ', '_') + '_initial.joblib'))
    rules = export_text(model.named_steps['classifier'], feature_names=list(model.named_steps['prepare'].get_feature_names_out()))
    (output / (name.lower().replace(' ', '_') + '_rules.txt')).write_text(rules)

print('Mean validation scores across five folds (not test scores):')
print(results.groupby('model')[metrics].mean().round(4).to_string())
print('\nCombined validation confusion counts:')
print(results.groupby('model')[['true_negative', 'false_positive', 'false_negative', 'true_positive']].sum().to_string())
print('\nThreshold: 0.5; max_depth=3, no class weighting or parameter tuning.')
print('Average precision summarizes the precision-recall curve; it is not trapezoidal PR-AUC.')
print('Each student predicted once by a model that did not train on their row.')
print('No test inputs or test answers were loaded.')
