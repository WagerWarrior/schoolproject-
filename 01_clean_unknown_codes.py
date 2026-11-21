"""Step 1: label documented unknowns, without dropping rows or guessing values.
Run with Python and pandas. No encoding, scaling, imputation or model fitting occurs.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / 'feature_plan.json').read_text())
source = (ROOT / plan['source_csv']).resolve()
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
raw = pd.read_csv(source)
mappings = json.loads((ROOT / 'resources/verified_code_mappings.json').read_text())
working = raw[plan['model_b_columns'] + ['Target']].copy()
# Values are codes from the assigned Kaggle CSV, not UCI's differently numbered codes.
rules = {"Mother's qualification": (19, 'Unknown'),
         "Father's qualification": (24, 'Unknown'),
         "Mother's occupation": (13, '(blank)'),
         "Father's occupation": (13, '(blank)')}
audit = []
for column, (code, meaning) in rules.items():
    assert mappings[column][str(code)]['meaning'] == meaning
    mask = raw[column].eq(code)
    # Decode the full column to readable category labels, avoiding mixed text/numbers.
    labels = {int(k): v['meaning'] for k, v in mappings[column].items()}
    assert set(raw[column].unique()).issubset(labels)
    working[column] = raw[column].map(labels)
    # The source's blank occupation category means unspecified; preserve that distinction.
    replacement = 'Unknown' if meaning == 'Unknown' else 'Not specified (source blank)'
    working.loc[mask, column] = replacement
    assert working[column].eq(replacement).sum() == mask.sum()
    audit.append({'column': column, 'original_code': code, 'source_meaning': meaning,
                  'replacement': replacement, 'affected_rows': int(mask.sum())})

assert len(working) == len(raw) == 4424
assert working['Target'].equals(raw['Target'])
assert not working.isna().any().any()
untouched = [c for c in working.columns if c not in rules]
pd.testing.assert_frame_equal(working[untouched], raw[untouched])
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
out = ROOT / 'data'
out.mkdir(exist_ok=True)
working.to_csv(out / 'selected_columns_step1.csv', index=False)
report = {'step': 1, 'status': 'Unknown categories labelled; not model-ready yet',
          'source_sha256': source_hash, 'rows_before': len(raw), 'rows_after': len(working),
          'model_a_inputs': len(plan['model_a_columns']), 'model_b_inputs': len(plan['model_b_columns']),
          'output_columns': len(working.columns), 'rules': audit,
          'notes': ['Four parental category columns decoded using verified dictionary labels.',
                    'Remaining category codes still require appropriate encoding.',
                    'No rows removed. No numerical values changed. Target remains separate.',
                    'No train/test split or fitted preprocessing yet.']}
(ROOT / 'step1_cleaning_audit.json').write_text(json.dumps(report, indent=2)+'\n')
for item in audit:
    print(f"{item['column']}: {item['original_code']} -> {item['replacement']} ({item['affected_rows']} rows)")
print(f'Saved {len(working)} rows, 17 selected inputs + original Target. Raw file unchanged.')
