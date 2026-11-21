"""Show standard scaling using training data only. No classifier trained yet."""
from pathlib import Path
import pandas as pd
from preprocessing import make_preprocessor

folder = Path(__file__).resolve().parent
training = pd.read_csv(folder / 'data/split/X_train.csv', index_col='source_row_index')

# Learn category names, numerical means and standard deviations from training.
# This is a demonstration. Cross-validation will fit a new preprocessor per fold.
preprocessor = make_preprocessor('B')
prepared = preprocessor.fit_transform(training)

scaler = preprocessor.named_transformers_['numbers']
numeric_columns = list(scaler.feature_names_in_)
scaled_numbers = pd.DataFrame(
    scaler.transform(training[numeric_columns]),
    columns=numeric_columns,
    index=training.index,
)

# Show five real students before and after scaling.
examples = pd.DataFrame(index=training.head(5).index)
for column in ['Age at enrollment', 'Curricular units 1st sem (grade)']:
    examples[column + ' - original'] = training[column].head(5)
    examples[column + ' - scaled'] = scaled_numbers[column].head(5)

# Save the learned numbers so the calculation is easy to follow.
settings = pd.DataFrame({
    'column': numeric_columns,
    'training_mean': scaler.mean_,
    'training_standard_deviation': scaler.scale_,
})
output = folder / 'analysis'
output.mkdir(exist_ok=True)
examples.round(4).to_csv(output / 'scaling_examples.csv')
settings.to_csv(output / 'scaling_settings_preview.csv', index=False)

print('Formula: (original value - training mean) / training standard deviation')
print(settings.round(3).to_string(index=False))
print('\nFive real training rows:')
print(examples.round(3).to_string())
print('\nPrepared shape:', prepared.shape)
print('Preview only. No test data loaded, no classifier fitted, original data unchanged.')
