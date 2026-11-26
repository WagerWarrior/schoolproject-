"""Create shared A/B train/test membership and training-only CV folds.
Run after 03_check_enrollment_categories.py. No model is trained here.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / 'feature_plan.json').read_text())
source = ROOT / 'data/selected_columns_step3.csv'
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
data = pd.read_csv(source)
assert data['Target'].isin(plan['target_mapping']).all()
# Map the recorded outcome to 1 for dropout and 0 for the other outcomes.
y = data['Target'].map(plan['target_mapping']).astype(int)
# Row positions track membership only; they are never model inputs.
# Keep similar dropout proportions in both groups; seed 42 makes the split repeatable.
train_ids, test_ids = train_test_split(
    data.index.to_numpy(), test_size=plan['split_plan']['test_fraction'],
    stratify=y, random_state=plan['split_plan']['random_state'])
assert not set(train_ids) & set(test_ids)
assert set(train_ids) | set(test_ids) == set(data.index)
out = ROOT / 'data/split'
out.mkdir(exist_ok=True)
manifest = pd.DataFrame({'source_row_index': data.index, 'split': 'train', 'validation_fold': -1})
manifest.loc[test_ids, 'split'] = 'test'
# Divide only the training group into five reusable validation groups.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (_, validation_positions) in enumerate(cv.split(train_ids, y.loc[train_ids]), start=1):
    manifest.loc[train_ids[validation_positions], 'validation_fold'] = fold
assert manifest.loc[train_ids, 'validation_fold'].between(1, 5).all()
assert manifest.loc[test_ids, 'validation_fold'].eq(-1).all()

report = {'source_sha256': source_hash, 'seed': 42,
          'stratify_on': 'binary dropout target', 'splits': {},
          'notes': ['source_row_index and validation_fold are metadata, never inputs.',
                    'Shared rows and validation folds for both A and B.',
                    'Test files reserved for final evaluation; no model or learned preprocessing fitted.',
                    'Earlier full-data audits checked validity and codes, not predictive performance.']}
# Save inputs and answers separately, keeping row numbers so they can be matched.
for name, ids in [('train', train_ids), ('test', test_ids)]:
    features = data.loc[ids, plan['model_b_columns']].copy()
    assert 'Target' not in features and not any('2nd sem' in c for c in features)
    features.index.name = 'source_row_index'
    features.to_csv(out / f'X_{name}.csv')
    labels = y.loc[ids].rename('dropout')
    labels.index.name = 'source_row_index'
    labels.to_csv(out / f'y_{name}.csv')
    saved = pd.read_csv(out / f'X_{name}.csv', index_col='source_row_index')
    pd.testing.assert_frame_equal(saved, features)
    report['splits'][name] = {'students': len(ids), 'dropout': int(labels.sum()),
                             'other_outcome': int(labels.eq(0).sum()),
                             'dropout_percent': round(float(labels.mean()*100), 2)}
manifest.to_csv(out / 'split_manifest.csv', index=False)
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
(ROOT / 'step4_split_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
