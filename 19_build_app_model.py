"""Rebuild the locked app artifact from training data only, without test evaluation."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from preprocessing import make_preprocessor

folder = Path(__file__).resolve().parent
destination = folder / 'models/final_logistic_b.joblib'
if destination.exists():
    raise SystemExit('The model already exists; no rebuild needed.')
config = json.loads((folder / 'final_model_config.json').read_text())
plan = json.loads((folder / 'feature_plan.json').read_text())
columns = plan['model_b_columns']
X = pd.read_csv(folder / 'data/split/X_train.csv', index_col='source_row_index')
y = pd.read_csv(folder / 'data/split/y_train.csv', index_col='source_row_index')['dropout']
assert X.index.equals(y.index)
model = Pipeline([
    ('prepare', make_preprocessor('B')),
    ('classifier', LogisticRegression(C=config['C'], max_iter=config['max_iter'])),
])
model.fit(X[columns], y)
destination.parent.mkdir(exist_ok=True)
joblib.dump({'pipeline': model, 'cutoff': config['cutoff'],
             'input_columns': columns, 'config': config}, destination)
print('Saved the locked app model using training data only. No test evaluation.')
