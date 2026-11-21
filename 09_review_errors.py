"""Draw confusion matrices from saved validation predictions, not test data."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

folder = Path(__file__).resolve().parent
predictions = pd.read_csv(folder / 'results/first_models_validation_predictions.csv')
names = ['Baseline', 'Logistic A', 'Logistic B']
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
for name, ax in zip(names, axes):
    rows = predictions[predictions['model'] == name]
    matrix = confusion_matrix(rows['actual'], rows['predicted'], labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=['Other outcome', 'Dropout'])
    display.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title(name)
fig.suptitle('Validation predictions: 3,539 students per model | cutoff 0.5', fontsize=13)
fig.savefig(folder / 'results/first_models_confusion_matrices.png', dpi=160)
plt.close(fig)
print('Saved validation confusion matrices. No training or test evaluation performed.')
