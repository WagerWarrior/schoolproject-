"""Check remaining enrollment fields and decode documented category names."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
source = ROOT / 'data/selected_columns_step1.csv'
original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
data = pd.read_csv(source)
working = data.copy()
mappings = json.loads((ROOT / 'resources/verified_code_mappings.json').read_text())
decoded = ['Course', 'Application mode', 'Previous qualification']
report = {'step': 3, 'rows': len(data), 'checks': {}, 'notes': []}
for column in decoded:
    labels = {int(k): v['meaning'] for k, v in mappings[column].items()}
    unexpected = sorted(set(data[column].dropna().unique()) - set(labels))
    assert not unexpected and not data[column].isna().any(), column
    working[column] = data[column].map(labels)
    report['checks'][column] = {
        'documented_categories_present': int(data[column].nunique()),
        'unmapped_codes': unexpected, 'missing': int(data[column].isna().sum()),
        'action': 'Replace codes with verified category labels; encode within training pipeline later.'}

for column in ['Daytime/evening attendance', 'Displaced']:
    assert data[column].isin([0, 1]).all(), column
    report['checks'][column] = {'counts': data[column].value_counts().sort_index().to_dict(),
                                'action': 'Keep documented binary 0/1 values.'}
order = data['Application order']
assert order.notna().all() and order.between(0, 9).all() and order.mod(1).eq(0).all()
report['checks']['Application order'] = {
    'counts': order.value_counts().sort_index().to_dict(),
    'action': 'Keep original integer preference order. Flag 0 and 9 for documentation review; do not guess corrections.',
    'review_note': 'Saved dictionary allows 0–9, but previously reviewed paper describes a minimum of 1. Preserve the single 0 pending clarification.'}
report['notes'] = [
    'Course means degree program, not number of courses.',
    'Application mode means admission route, not online versus paper submission.',
    'Previous qualification category codes are not numerical education scores.',
    'Daytime/evening attendance: 1 daytime, 0 evening.',
    'Displaced: 1 yes, 0 no; studying away from usual residence.',
    'Rare admission categories retained. Any grouping learned from frequencies must use training data only.',
    'No rows removed, no imputation or model fitting. Target unchanged.'
]
pd.testing.assert_frame_equal(working.drop(columns=decoded), data.drop(columns=decoded))
assert not working.isna().any().any()
assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
working.to_csv(ROOT / 'data/selected_columns_step3.csv', index=False)
(ROOT / 'step3_category_audit.json').write_text(json.dumps(report, indent=2) + '\n')
log = ROOT / 'Project Decision Log.txt'
entry = '''

2026-11-21: Cleaning steps 2 and 3
Checked age and six first-semester numeric columns: no missing, negative,
fractional count or out-of-range grade values. Cross-column checks found no
approved counts exceeding enrolled counts. Retained all zeros and all rows.
Checked remaining six enrollment fields. Course (17 categories), application
mode (18) and previous qualification (17) all match verified code mappings;
decoded these three columns to readable labels. Binary attendance and displaced
fields contain only 0/1. Application order retained; its single 0 remains a
documented review item due to differing source descriptions. No guessed fix.
Current working data: data/selected_columns_step3.csv, 4424 rows, 17 inputs
plus original Target. No fitted encoding, train/test split or modelling yet.
'''
existing = log.read_text()
if '2026-11-21: Cleaning steps 2 and 3' not in existing:
    log.write_text(existing + entry)
print(json.dumps(report, indent=2))
