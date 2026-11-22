"""Simple classroom dashboard. Run: streamlit run app.py"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

folder = Path(__file__).resolve().parent
st.set_page_config(page_title='Student dropout project', page_icon=':material/school:', layout='wide')
st.html((folder / 'app.css').read_text())
st.sidebar.title('Student dropout')
st.sidebar.caption('Explore the evidence. Try an example.')
st.title('Student dropout project')
st.caption('Does first-semester progress help us identify recorded dropout outcomes?')
st.caption('Educational demonstration using one Portuguese institution. Use made-up profiles only. '
        'Do not use this app for real admission, funding or student-support decisions.')
page = st.sidebar.radio('Explore', ['Results', 'What mattered', 'Try a made-up profile'])

if page == 'Results':
    st.header('Final model: enrollment + first semester')
    report = json.loads((folder / 'results/final_test_metrics.json').read_text())
    st.write('Logistic regression, C = 1, cutoff = 0.3. Trained on 3,539 students '
             'and evaluated once on 885 reserved test students.')
    columns = st.columns(4)
    for column, label, key in zip(columns, ['Accuracy', 'Precision', 'Recall', 'F1'],
                                   ['accuracy', 'precision', 'recall', 'F1']):
        column.metric(label, f"{report[key]:.3f}" if key == 'F1' else f"{report[key]:.1%}")
    st.write('Recall: how many actual dropouts we caught. Precision: how many flagged '
             'students actually had a dropout outcome. F1 combines precision and recall.')
    st.subheader('Correct predictions and mistakes')
    matrix = pd.DataFrame(
        [[report['true_negative'], report['false_positive']],
         [report['false_negative'], report['true_positive']]],
        index=['Actual enrolled / graduated', 'Actual dropout'],
        columns=['Predicted other outcome', 'Predicted dropout'])
    st.dataframe(matrix)
    st.caption('100 false alarms; 68 missed dropouts. Still enrolled does not guarantee graduation.')
    st.divider()
    st.subheader('Compare A and B during validation')
    st.write('A uses enrollment inputs; B adds first-semester progress. These are the '
             'initial models at cutoff 0.5, before tuning. They are validation results, not test results.')
    comparison = pd.concat([pd.read_csv(folder / 'results' / filename) for filename in
                            ['first_models_folds.csv', 'decision_tree_folds.csv', 'random_forest_folds.csv']])
    view = comparison.groupby('model')[['accuracy', 'precision', 'recall', 'F1']].mean()
    selected = st.multiselect('Models to compare', view.index.tolist(), default=view.index.tolist())
    if selected:
        st.dataframe(view.loc[selected].round(3))
        st.bar_chart(view.loc[selected, ['F1']])
    else:
        st.info('Choose at least one model to see the comparison.')
    st.caption('Final B was selected using validation. Only B received the final test evaluation.')
    with st.expander('More test metrics and limitations'):
        st.write(f"ROC-AUC: {report['ROC_AUC']:.3f} · Average precision: "
                 f"{report['average_precision']:.3f} · PR-AUC (trapezoidal): {report['PR_AUC_trapezoidal']:.3f}")
        st.write('No dropout dates are available, so this is retrospective classification, '
                 'not a validated prediction of future dropout after semester one. '
                 'Results may not generalize to other institutions. Parental information '
                 'may reflect socioeconomic differences; subgroup fairness review remains needed.')

elif page == 'What mattered':
    st.header('Which columns did the model rely on?')
    st.write('We shuffled one column at a time in validation data. A larger drop in '
             'average precision suggests the model relied more on that information.')
    importance = pd.read_csv(folder / 'results/column_importance_summary.csv', index_col='column')
    count = st.slider('Number of columns to show', 3, 17, 6)
    top = importance.head(count).rename(columns={'mean': 'Average precision drop', 'std': 'Variation'})
    st.bar_chart(top[['Average precision drop']], horizontal=True, sort=False, color='#245b78')
    st.dataframe(top.round(4))
    st.write('Courses passed stood out most. Its negative model weight means more passed '
             'courses lowers the dropout score when the other inputs stay fixed.')
    st.caption('These are metric decreases, not percentages of total importance. '
               'Related columns share information, and shuffling can create unrealistic '
               'combinations. This explains model behavior, not causes of dropout.')

else:
    st.header('Try a made-up student profile')
    st.write('Meet Alex, a fictional 20-year-old nursing student. The form starts with our example values; you can change them.')
    st.caption('This form uses the locked final model. Inputs are not saved to a file or database.')
    with st.expander('How to follow this example', expanded=True):
        st.markdown('**1. Review the profile** — Alex studies during the day and lives at their usual residence.\n\n'
                    '**2. Review semester one** — 6 courses enrolled, 4 passed, average grade 12 out of 20.\n\n'
                    '**3. Calculate** — compare the model score with our fixed 0.3 cutoff.')
    artifact_path = folder / 'models/final_logistic_b.joblib'
    if not artifact_path.exists():
        st.warning('The local model file is missing. Follow the app setup instructions in README.md.')
        st.stop()
    artifact = joblib.load(artifact_path)
    model = artifact['pipeline']
    encoder = model.named_steps['prepare'].named_transformers_['categories']
    category_columns = list(encoder.feature_names_in_)
    labels = {
        'Course': 'Degree program', 'Application mode': 'Admission route',
        'Previous qualification': 'Previous education',
        'Displaced': 'Studying away from usual residence',
    }
    defaults = {'Course': 'Nursing', 'Previous qualification': 'Secondary education',
                "Mother's qualification": 'Unknown', "Father's qualification": 'Unknown',
                "Mother's occupation": 'Not specified (source blank)',
                "Father's occupation": 'Not specified (source blank)'}
    profile = {}
    with st.form('profile'):
        st.subheader('1. Enrollment information')
        left, right = st.columns(2)
        for i, (column, choices) in enumerate(zip(category_columns, encoder.categories_)):
            options = list(choices)
            default = defaults.get(column, options[0])
            index = options.index(default) if default in options else 0
            with (left if i % 2 == 0 else right):
                profile[column] = st.selectbox(labels.get(column, column), options, index=index, key=column)
        left, right = st.columns(2)
        profile['Age at enrollment'] = left.number_input('Age at enrollment', 17, 100, 20)
        profile['Application order'] = right.number_input('Application preference order', 0, 9, 1,
                                                     help='Source codes are retained; order 0 has a documented ambiguity.')
        profile['Daytime/evening attendance'] = int(left.selectbox('Attendance', ['Daytime', 'Evening']) == 'Daytime')
        profile['Displaced'] = int(right.selectbox(labels['Displaced'], ['No', 'Yes']) == 'Yes')
        st.divider()
        st.subheader('2. First-semester results')
        left, right = st.columns(2)
        prefix = 'Curricular units 1st sem '
        fields = [('credited', 'Courses credited from previous learning', 0),
                  ('enrolled', 'Courses enrolled', 6), ('evaluations', 'Assessment attempts', 6),
                  ('approved', 'Courses passed', 4), ('without evaluations', 'Courses without assessments', 0)]
        for i, (suffix, label, default) in enumerate(fields):
            profile[prefix + '(' + suffix + ')'] = (left if i % 2 == 0 else right).number_input(label, 0, 100, default)
        profile[prefix + '(grade)'] = right.number_input('Average grade (out of 20)', 0.0, 20.0, 12.0, step=0.1)
        submitted = st.form_submit_button('Calculate example score', type='primary', use_container_width=True)
    if submitted:
        enrolled = profile[prefix + '(enrolled)']
        if any(profile[prefix + '(' + name + ')'] > enrolled for name in ['approved', 'credited', 'without evaluations']):
            st.error('Courses passed, credited, or without assessments cannot exceed courses enrolled in this demo.')
        else:
            row = pd.DataFrame([profile])[artifact['input_columns']]
            score = float(model.predict_proba(row)[0, 1])
            cutoff = artifact['cutoff']
            st.divider()
            st.subheader('3. Read the result')
            st.metric('Example model score', f'{score:.3f}')
            st.write(f"Cutoff: {cutoff:.1f}. Classification: **"
                     f"{'Dropout' if score >= cutoff else 'Other outcome'}**.")
            st.write(f"The model converted the profile into numbers, applied its learned weights, "
                     f"and returned **{score:.3f}**. Because this is "
                     f"{'at or above' if score >= cutoff else 'below'} **{cutoff:.1f}**, "
                     f"it assigns the {'dropout' if score >= cutoff else 'other outcome'} class.")
            st.info('Try changing only Courses passed from 4 to 6, then calculate again. '
                    'This shows how the model reacts when the other inputs stay fixed; it does not prove cause and effect.')
            st.caption('This score is not a confirmed probability. It is a classroom example, '
                       'not a prediction about a real student. Scores for unusual profiles may be unreliable.')
