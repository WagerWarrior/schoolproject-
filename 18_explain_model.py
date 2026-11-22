"""Explain weights and original-column importance without using test rows."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from preprocessing import make_preprocessor

folder = Path(__file__).resolve().parent
config = json.loads((folder / 'final_model_config.json').read_text())
artifact = joblib.load(folder / 'models/final_logistic_b.joblib')
model = artifact['pipeline']
output = folder / 'results'

# A positive coefficient pushes the dropout score up; a negative one pushes down.
# Numeric weights apply to one standard deviation, because we scaled inputs.
names = model.named_steps['prepare'].get_feature_names_out()
weights = pd.DataFrame({
    'encoded_feature': names,
    'weight': model.named_steps['classifier'].coef_[0],
})
weights['absolute_weight'] = weights['weight'].abs()
weights.sort_values('absolute_weight', ascending=False).to_csv(
    output / 'final_model_weights.csv', index=False)
numeric = weights[weights['encoded_feature'].str.startswith('numbers__')].copy()
numeric['column'] = numeric['encoded_feature'].str.replace('numbers__', '', regex=False)
numeric.sort_values('absolute_weight', ascending=False).to_csv(
    output / 'final_numeric_weights.csv', index=False)

# Shuffle one original column in validation rows to see how much it helps.
# This is interpretation of our locked setup, not a new feature-selection step.
X = pd.read_csv(folder / 'data/split/X_train.csv', index_col='source_row_index')
y = pd.read_csv(folder / 'data/split/y_train.csv', index_col='source_row_index')['dropout']
X = X[artifact['input_columns']]
manifest = pd.read_csv(folder / 'data/split/split_manifest.csv', index_col='source_row_index')
folds = manifest.loc[X.index, 'validation_fold']
assert X.index.equals(y.index) and folds.between(1, 5).all()
rows = []
for fold in range(1, 6):
    train_rows = folds != fold
    validation_rows = folds == fold
    fold_model = Pipeline([
        ('prepare', make_preprocessor('B')),
        ('classifier', LogisticRegression(C=config['C'], max_iter=config['max_iter'])),
    ])
    fold_model.fit(X.loc[train_rows], y.loc[train_rows])
    # Average precision evaluates the score ranking without changing our cutoff.
    importance = permutation_importance(
        fold_model, X.loc[validation_rows], y.loc[validation_rows],
        scoring='average_precision', n_repeats=5, random_state=42,
    )
    for column, values in zip(X.columns, importance.importances):
        for repeat, value in enumerate(values, start=1):
            rows.append({'column': column, 'fold': fold, 'repeat': repeat,
                         'average_precision_drop': value})

details = pd.DataFrame(rows)
details.to_csv(output / 'column_importance_repeats.csv', index=False)
summary = details.groupby('column')['average_precision_drop'].agg(['mean', 'std'])
summary = summary.sort_values('mean', ascending=False)
summary.to_csv(output / 'column_importance_summary.csv')
print('Validation average precision decrease after shuffling each column:')
print(summary.round(4).to_string())
print('\nFinal numeric weights (per one training standard deviation):')
print(numeric.sort_values('absolute_weight', ascending=False)[['column', 'weight']].round(4).to_string(index=False))
print('No test rows loaded, no features removed, no final model overwritten.')
print('Correlated inputs can share importance; weights and shuffling do not prove causation.')
