"""Review subgroup errors using saved validation scores, without training or testing again."""
from pathlib import Path
import json
import pandas as pd

folder = Path(__file__).resolve().parent
config = json.loads((folder / 'final_model_config.json').read_text())
plan = json.loads((folder / 'feature_plan.json').read_text())
raw = pd.read_csv(folder / plan['source_csv'])
manifest = pd.read_csv(folder / 'data/split/split_manifest.csv', index_col='source_row_index')
answers = pd.read_csv(folder / 'data/split/y_train.csv', index_col='source_row_index')['dropout']
saved = pd.read_csv(folder / 'results/logistic_c_validation_predictions.csv')

# Select the locked B setting. Each student was predicted by a fold model that did not train on them.
rows = saved[(saved['version'] == 'B') & (saved['C'] == config['C'])].copy()
rows = rows.set_index('source_row_index')
assert config['feature_set'] == 'B' and config['cutoff'] == 0.3
assert rows.index.is_unique and set(rows.index) == set(answers.index)
assert manifest.loc[rows.index, 'split'].eq('train').all()
assert (manifest.loc[rows.index, 'validation_fold'] == rows['fold']).all()
assert answers.loc[rows.index].equals(rows['actual'].rename('dropout'))
assert raw.loc[rows.index, 'Target'].map(plan['target_mapping']).equals(rows['actual'].rename('Target'))
assert rows['score'].between(0, 1).all()

# Reapply 0.3 because the saved prediction labels used the earlier 0.5 cutoff.
rows['predicted'] = (rows['score'] >= config['cutoff']).astype(int)
fields = ['Gender', 'International', 'Educational special needs', 'Marital status', 'Nacionality']
mappings = json.loads((folder / 'resources/verified_code_mappings.json').read_text())

# Join only training-row attributes. These audit fields do not become model inputs.
for field in fields:
    values = raw.loc[rows.index, field]
    if field in mappings:
        labels = {int(code): item['meaning'] for code, item in mappings[field].items()}
        assert values.isin(labels).all()
        rows[field] = values.map(labels)
    else:
        # Preserve source codes rather than guessing category meanings.
        rows[field] = values.map(lambda value: 'Source code ' + str(value))
assert not rows[fields].isna().any().any()

# These are cautious reporting rules, not a statistical guarantee or a privacy guarantee.
minimum_group = 30
minimum_denominator = 20

def rate(numerator, denominator):
    """Leave a rate blank when too few students support its denominator."""
    if denominator < minimum_denominator:
        return None
    return numerator / denominator

results = []
for field in fields:
    for group, part in rows.groupby(field):
        item = {'field': field, 'group': group, 'students': len(part)}
        if len(part) < minimum_group:
            item['status'] = 'Too few students; outcomes and rates withheld'
        else:
            # Count the four possible combinations of actual and predicted outcome.
            actual = part['actual']
            predicted = part['predicted']
            tp = int(((actual == 1) & (predicted == 1)).sum())
            fn = int(((actual == 1) & (predicted == 0)).sum())
            fp = int(((actual == 0) & (predicted == 1)).sum())
            tn = int(((actual == 0) & (predicted == 0)).sum())
            assert tp + fn + fp + tn == len(part)
            item.update({'actual_dropout': tp + fn, 'actual_other': tn + fp,
                         'caught_dropout': tp, 'missed_dropout': fn,
                         'false_alarm': fp, 'correct_other': tn,
                         'recall': rate(tp, tp + fn),
                         'false_negative_rate': rate(fn, tp + fn),
                         'false_positive_rate': rate(fp, fp + tn),
                         'precision': rate(tp, tp + fp),
                         'status': 'Descriptive only; rates with denominator below 20 withheld'})
        results.append(item)

# Publish group summaries only, never the joined student-level audit table.
summary = pd.DataFrame(results)
for field in fields:
    assert summary.loc[summary['field'] == field, 'students'].sum() == len(rows)
summary.to_csv(folder / 'results/subgroup_validation_summary.csv', index=False)
print(summary.to_string(index=False))
print('Training-only validation; selected setting; no independent fairness certification.')
print('Groups overlap across fields. Do not add their student counts across different fields.')
