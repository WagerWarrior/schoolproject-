"""Descriptive EDA using only the saved training inputs and answers."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'analysis'
OUT.mkdir(exist_ok=True)
X = pd.read_csv(ROOT / 'data/split/X_train.csv', index_col='source_row_index')
y = pd.read_csv(ROOT / 'data/split/y_train.csv', index_col='source_row_index')['dropout']
assert X.index.equals(y.index) and X.index.is_unique
assert 'Target' not in X and 'dropout' not in X
assert y.isin([0, 1]).all()
grade = 'Curricular units 1st sem (grade)'
passed = 'Curricular units 1st sem (approved)'
enrolled = 'Curricular units 1st sem (enrolled)'
joined = X.join(y)
numeric = ['Age at enrollment', 'Application order'] + [c for c in X if '1st sem' in c]
X[numeric].describe().T.to_csv(OUT / 'training_numeric_summary.csv')
categories = [c for c in X if c not in numeric]
pd.concat([X[c].value_counts(dropna=False).rename_axis('category').reset_index(name='students').assign(column=c)
           for c in categories], ignore_index=True).to_csv(OUT / 'training_category_counts.csv', index=False)
comparison = joined.groupby('dropout')[[grade, passed, enrolled]].agg(['mean', 'median'])
comparison.to_csv(OUT / 'training_outcome_comparison.csv')
unknowns = {c: int(X[c].isin(['Unknown', 'Not specified (source blank)']).sum()) for c in categories}
report = {
    'population': 'Training data only; no test data loaded', 'students': len(X),
    'inputs': len(X.columns), 'missing_cells': int(X.isna().sum().sum()),
    'explicit_unknown_or_unspecified_counts': {c: n for c, n in unknowns.items() if n},
    'dropouts': int(y.sum()), 'other_outcomes': int(y.eq(0).sum()),
    'majority_baseline_training_accuracy_percent': round(float(y.eq(0).mean()*100), 2),
    'zero_grade_rows': int(X[grade].eq(0).sum()),
    'zero_enrolled_rows': int(X[enrolled].eq(0).sum()),
    'age_median': float(X['Age at enrollment'].median()),
    'grade_medians_by_outcome': {str(k): float(v) for k,v in joined.groupby('dropout')[grade].median().items()},
    'passed_medians_by_outcome': {str(k): float(v) for k,v in joined.groupby('dropout')[passed].median().items()},
    'decisions': [
        'No rows removed, no values replaced, and no models fitted during EDA.',
        'No blank-cell imputation currently needed; explicit unknown categories remain meaningful categories.',
        'Retain zero grades. Describe their concentration instead of replacing them with the mean.',
        'Always predicting other outcome misses every dropout despite about 68% training accuracy.',
        'Grade differences are descriptive associations, not causal effects or model performance.',
        'Rare categories remain; use an encoder that handles categories absent from a training fold.',
        'Keep 17-input starting plan. Cross-validation, not these summaries, determines model choices.'
    ]}
(OUT / 'training_eda_summary.json').write_text(json.dumps(report, indent=2)+'\n')
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
colors = ['#245b78', '#d56a36']
axes[0].bar(['Other outcome', 'Dropout'], [y.eq(0).sum(), y.sum()], color=colors)
axes[0].set(title='Recorded outcomes', ylabel='Students')
for i, count in enumerate([int(y.eq(0).sum()), int(y.sum())]):
    axes[0].text(i, count+35, f'{count:,}', ha='center')
axes[0].set_ylim(0, 2750)
axes[1].hist(X['Age at enrollment'], bins=range(15, 66, 5), color=colors[0], edgecolor='white')
axes[1].set(title='Age when enrolling', xlabel='Age (years)', ylabel='Students')
for outcome, label, color in [(0,'Other outcome',colors[0]), (1,'Dropout',colors[1])]:
    values = X.loc[y.eq(outcome), grade]
    axes[2].hist(values, bins=range(0, 22, 2), weights=[100/len(values)]*len(values),
                 histtype='step', linewidth=2, label=label, color=color)
axes[2].set(title='First-semester grades', xlabel='Recorded grade (out of 20)', ylabel='% within outcome group')
axes[2].legend(frameon=False)
fig.suptitle('Training data exploration — 3,539 students', fontsize=16)
fig.savefig(OUT / 'training_overview.png', dpi=160)
plt.close(fig)
print(json.dumps(report, indent=2))
