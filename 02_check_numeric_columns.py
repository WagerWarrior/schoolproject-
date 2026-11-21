"""Step 2: audit numerical inputs without guessing corrections or using outcomes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
source = ROOT / 'data/selected_columns_step1.csv'
before = hashlib.sha256(source.read_bytes()).hexdigest()
data = pd.read_csv(source)
prefix = 'Curricular units 1st sem '
age = 'Age at enrollment'
grade = prefix + '(grade)'
counts = [prefix + '(' + name + ')' for name in
          ['credited', 'enrolled', 'evaluations', 'approved', 'without evaluations']]
report = {'step': 2, 'rows': len(data), 'columns': {}, 'consistency_checks': {},
          'decision': 'Preserve all values and rows; no confirmed corrections required by these checks.'}
for column in [age] + counts + [grade]:
    values = pd.to_numeric(data[column], errors='coerce')
    checks = {
        'missing_or_not_numeric': int(values.isna().sum()),
        'nonfinite_nonmissing': int((values.notna() & ~np.isfinite(values)).sum()),
        'negative': int(values.lt(0).sum()),
    }
    if column != grade:
        checks['fractional_values'] = int((values.notna() & values.mod(1).ne(0)).sum())
    if column == grade:
        checks['outside_0_to_20'] = int((values.lt(0) | values.gt(20)).sum())
    report['columns'][column] = {
        'minimum': float(values.min()), 'maximum': float(values.max()),
        'zeros': int(values.eq(0).sum()), 'checks': checks}

enrolled = data[prefix + '(enrolled)']
approved = data[prefix + '(approved)']
report['consistency_checks'] = {
    'approved_greater_than_enrolled': int(approved.gt(enrolled).sum()),
    'without_evaluations_greater_than_enrolled': int(data[prefix + '(without evaluations)'].gt(enrolled).sum()),
    'credited_greater_than_enrolled_review_only': int(data[prefix + '(credited)'].gt(enrolled).sum()),
    'zero_grade_with_positive_approved_review_only': int((data[grade].eq(0) & approved.gt(0)).sum()),
    'positive_grade_with_zero_approved_review_only': int((data[grade].gt(0) & approved.eq(0)).sum()),
}
report['notes'] = [
    'Age range is descriptive; older students are not removed merely for being unusual.',
    'Zero grades are retained. Do not automatically replace them with an average or call them missing.',
    'Evaluation attempts may exceed enrolled units, so this is not treated as an error.',
    'Enrolled minus approved does not establish the number of failed courses.',
    'Cross-column flags require source review before any correction.',
    'No target-based exploration, fitted preprocessing, scaling, imputation or row removal performed.',
    'The working data remains selected_columns_step1.csv; no duplicate unchanged data file created.'
]
assert hashlib.sha256(source.read_bytes()).hexdigest() == before
(ROOT / 'step2_numeric_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
