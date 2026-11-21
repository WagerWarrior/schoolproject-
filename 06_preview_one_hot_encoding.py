"""Training-only teaching preview. Do NOT use this fitted preview for CV."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from preprocessing import make_encoder, CATEGORY_COLUMNS

ROOT = Path(__file__).resolve().parent
X = pd.read_csv(ROOT / 'data/split/X_train.csv', index_col='source_row_index')
out = ROOT / 'analysis'
out.mkdir(exist_ok=True)
report = {'purpose': 'Training-only preview, not a fitted preprocessing artifact for model validation',
          'categorical_inputs': CATEGORY_COLUMNS, 'models': {}}
for model in ('A', 'B'):
    encoder = make_encoder(model)
    encoded = encoder.fit_transform(X)
    names = encoder.get_feature_names_out()
    assert encoded.shape[0] == len(X) and encoded.shape[1] == len(names)
    fitted = encoder.named_transformers_['categories']
    categorical = fitted.transform(X[CATEGORY_COLUMNS])
    assert np.isin(categorical.data, [0, 1]).all()
    assert np.all(np.asarray(categorical.sum(axis=1)).ravel() == len(CATEGORY_COLUMNS))
    report['models'][model] = {'rows': len(X), 'encoded_columns': len(names)}
    if model == 'B':
        report['category_counts'] = {c: len(values) for c, values in zip(CATEGORY_COLUMNS, fitted.categories_)}
        # Verify a category absent from fitting does not crash transformation.
        unseen = X.iloc[[0]].copy()
        unseen['Course'] = '__UNSEEN_PROGRAM_FOR_CHECK__'
        checked = fitted.transform(unseen[CATEGORY_COLUMNS]).toarray()
        assert checked[0, :len(fitted.categories_[0])].sum() == 0
        report['unseen_category_handling'] = 'All zeros for that field; distinct from a documented Unknown category.'
        # Show one real training row from each of three familiar programs.
        examples = pd.concat([X[X['Course'].eq(c)].head(1) for c in ['Nursing', 'Management', 'Tourism']])
        transformed = encoder.transform(examples).toarray()
        preview = pd.DataFrame(transformed, index=examples.index, columns=names)
        chosen = ['categories__Course_'+c for c in ['Nursing', 'Management', 'Tourism']]
        preview = preview[chosen].astype(int)
        preview.columns = ['Nursing', 'Management', 'Tourism']
        preview.insert(0, 'Original program', examples['Course'])
        preview.to_csv(out / 'one_hot_program_examples.csv')
        print(preview.to_string(index=False))
report['notes'] = [
    'Seven category inputs expand to separate 0/1 indicator columns.',
    'Age, application order, binary fields and first-semester numbers are unchanged here.',
    'Target and source row index are never model inputs.',
    'No test data loaded. No classifier trained. No scaling fitted.',
    'Preview encoder is discarded. Future model pipelines must call make_encoder afresh inside CV.',
    'Encoded column counts can vary across training folds if rare categories are absent.'
]
(out / 'one_hot_encoding_audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
