"""Smoke-check page navigation, the saved cutoff and invalid-input handling."""
from pathlib import Path
import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

folder = Path(__file__).resolve().parent
# Run the app without a browser, then interact with its controls in this check.
app = AppTest.from_file(str(folder / 'app.py')).run(timeout=30)
assert not app.exception
assert [item.value for item in app.metric] == ['81.0%', '68.4%', '76.1%', '0.720']
app.multiselect[0].set_value(['Logistic A', 'Logistic B']).run()
assert not app.exception
app.multiselect[0].set_value([]).run()
assert not app.exception and app.info[0].value == 'Choose at least one model to see the comparison.'
app.sidebar.radio[0].set_value('What mattered').run()
app.slider[0].set_value(17).run()
assert not app.exception
app.sidebar.radio[0].set_value('Try a made-up profile').run()
assert not app.exception

# Compare the form's prediction to a direct call of the saved pipeline.
artifact = joblib.load(folder / 'models/final_logistic_b.joblib')
encoder = artifact['pipeline'].named_steps['prepare'].named_transformers_['categories']
profile = {column: app.selectbox(key=column).value for column in encoder.feature_names_in_}
numeric_columns = ['Age at enrollment', 'Application order',
                   'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
                   'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
                   'Curricular units 1st sem (without evaluations)', 'Curricular units 1st sem (grade)']
# Match inputs by their displayed labels so moving fields does not break the check.
labels = ['Age at enrollment', 'Application preference order', 'Courses credited from previous learning',
          'Courses enrolled', 'Assessment attempts', 'Courses passed', 'Courses without assessments',
          'Average grade (out of 20)']
widgets = {widget.label: widget for widget in app.number_input}
profile.update({column: widgets[label].value for column, label in zip(numeric_columns, labels)})
profile['Daytime/evening attendance'] = 1
profile['Displaced'] = 0
expected = float(artifact['pipeline'].predict_proba(pd.DataFrame([profile])[artifact['input_columns']])[0, 1])
app.button[0].click().run()
assert not app.exception and not app.error
assert app.metric[0].value == f'{expected:.3f}'
assert 0.3 < expected < 0.5  # Distinguishes our locked cutoff from the default 0.5.
assert any('**Dropout**' in item.value for item in app.markdown)

print('Alex baseline:', expected)
# Change only courses passed and verify that the app still matches the saved pipeline.
next(w for w in app.number_input if w.label == 'Courses passed').set_value(6)
app.button[0].click().run()
assert not app.exception and not app.error
profile['Curricular units 1st sem (approved)'] = 6
improved = float(artifact['pipeline'].predict_proba(pd.DataFrame([profile])[artifact['input_columns']])[0, 1])
assert app.metric[0].value == f'{improved:.3f}'
print('Alex with six courses passed:', improved)

# Impossible course counts should produce an error, with no prediction shown.
next(w for w in app.number_input if w.label == 'Courses passed').set_value(7)
app.button[0].click().run()
assert not app.exception and app.error and not app.metric
print('Passed: pages, filters, score matches saved pipeline, cutoff 0.3, invalid course counts.')
