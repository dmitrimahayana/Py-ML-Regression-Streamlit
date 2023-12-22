from joblib import load
from sklearn.metrics import accuracy_score
import pandas as pd
import streamlit as st


def predict(model_name: str, df_input: pd.DataFrame):
    # Define the columns
    target_feature = 'New Size'

    # Define x and y
    x_test = df_input.drop(target_feature, axis=1)
    y_test = df_input[target_feature]

    # Load model from local folder
    model_folder = f"Model/Glamorise Estimated Weight/"
    model_filename = f"{model_name}.joblib"
    loaded_model = load(model_folder + model_filename)

    # Predict data
    y_pred = loaded_model.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    accuracy = loaded_model.score(x_test, y_test)

    # Convert prediction into dataframe
    df_y_pred = pd.DataFrame(y_pred, columns=['Predictions'])

    # Reset index on x_test to align with y_pred_df if needed
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Concatenate column-wise
    df_combined = pd.concat([x_test, y_test, df_y_pred], axis=1)

    return df_combined, accuracy


models = [
    'RandomForest',
    'KNN',
    'DecisionTree',
    'LogisticRegression',
    'SVM',
]

st.header('Glamorise Weight Dataset :sunglasses:', divider='rainbow')
with st.form("my_form"):
    st.header("Insert Your Own Data :test_tube:")
    option_model = st.selectbox(
        'Model Name',
        (model for model in models))
    number_weight = st.number_input('Unit Weight in grams', min_value=59)
    option_underwire = st.selectbox(
        'Underwire',
        ('No', 'Yes'))
    option_sport = st.selectbox(
        'Sports',
        ('No', 'Yes'))
    option_opening = st.selectbox(
        'Front Opening',
        ('No', 'Yes'))

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        dict_input = {
            'Unit Weight in grams': [number_weight],
            'Underwire': [option_underwire],
            'Sports': [option_sport],
            'Front Opening': [option_opening],
            'New Size': [''],
        }
        df_input2 = pd.DataFrame(data=dict_input)
        df_pred2, accuracy2 = predict(option_model, df_input2)
        st.subheader(f'This is a result using model _{option_model}_')
        st.subheader(f"Prediction Size: :blue[{df_pred2['Predictions'].values[0]}]")
