"""Factories return NEW, unfitted preprocessing for use inside model pipelines."""
import json
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
CATEGORY_COLUMNS = [
    'Course', 'Application mode', 'Previous qualification',
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]

def make_encoder(model='B'):
    """Encode nominal inputs; preserve numeric and binary fields unchanged.

    Fit this inside each CV training fold, never before cross-validation.
    This encoding-only version is retained for the step 6 demonstration.
    """
    if model not in ('A', 'B'):
        raise ValueError('model must be A or B')
    plan = json.loads((ROOT / 'feature_plan.json').read_text())
    selected = plan[f'model_{model.lower()}_columns']
    other = [c for c in selected if c not in CATEGORY_COLUMNS]
    return ColumnTransformer([
        ('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=True), CATEGORY_COLUMNS),
        ('numeric_and_binary', 'passthrough', other),
    ], remainder='drop')


def make_preprocessor(model='B'):
    """Prepare inputs for logistic regression. Fit inside each training fold."""
    if model not in ('A', 'B'):
        raise ValueError('model must be A or B')

    plan = json.loads((ROOT / 'feature_plan.json').read_text())
    selected = plan[f'model_{model.lower()}_columns']
    binary_columns = ['Daytime/evening attendance', 'Displaced']
    numeric_columns = [column for column in selected
                       if column not in CATEGORY_COLUMNS + binary_columns]

    # Each group of columns gets the preparation it needs.
    return ColumnTransformer([
        ('categories', OneHotEncoder(handle_unknown='ignore'), CATEGORY_COLUMNS),
        ('numbers', StandardScaler(), numeric_columns),
        ('binary', 'passthrough', binary_columns),
    ], remainder='drop')
