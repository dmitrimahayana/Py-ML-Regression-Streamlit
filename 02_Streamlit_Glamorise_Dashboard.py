from joblib import load
from sklearn.metrics import accuracy_score
import pandas as pd
import streamlit as st


def cleanse_data(df_input: pd.DataFrame):
    # Select specific column
    df_input = df_input[['Unit Weight in grams', 'Underwire', 'Sports', 'Front Opening', 'Size', 'Cup']]

    # Trim value
    df_input.loc[:, 'Underwire'] = df_input['Underwire'].str.strip()
    df_input.loc[:, 'Sports'] = df_input['Sports'].str.strip()
    df_input.loc[:, 'Front Opening'] = df_input['Front Opening'].str.strip()

    # Remove rows where any of the specified columns have empty/null values
    df_input = df_input.dropna(subset=['Unit Weight in grams', 'Size', 'Cup'])

    # Remove Unit Weight in grams with value 0
    df_input = df_input[df_input["Unit Weight in grams"] != 0]

    # Convert columns into integer
    df_input['Unit Weight in grams'] = df_input['Unit Weight in grams'].astype(int)
    df_input['Size'] = df_input['Size'].astype(int)

    # Combine size and cup
    df_input['New Size'] = df_input['Size'].astype(str) + df_input['Cup'].astype(str)

    # Drop size and cup columns
    df_input = df_input.drop('Size', axis=1)
    df_input = df_input.drop('Cup', axis=1)

    return df_input


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
    accuracy = accuracy_score(y_test, y_pred)

    # Convert prediction into dataframe
    df_y_pred = pd.DataFrame(y_pred, columns=['Predictions'])

    # Reset index on x_test to align with y_pred_df if needed
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Concatenate column-wise
    df_combined = pd.concat([x_test, y_test, df_y_pred], axis=1)

    return df_combined, accuracy


with st.sidebar:
    st.link_button("Go to EDA", "https://streamlit.io/gallery")
    st.link_button("Go to Machine Learning", "https://streamlit.io/gallery")

# Define dataset
filename = "Glamorise Estimated Weight from Supplier.xlsx"
file_path = f"Dataset/{filename}"
df_input = pd.read_excel(open(file_path, 'rb'), skiprows=1, sheet_name='Sheet1')
df_input = cleanse_data(df_input)

models = [
    'LogisticRegression',
    'DecisionTree',
    'RandomForest',
    'SVM',
    'KNN'
]
st.title('Glamorise Weight Dataset :sunglasses:')
for model in models:
    st.subheader(f'This is a prediction result using model _{model}_')
    df_pred, accuracy = predict(model, df_input)
    st.subheader(f'Accuracy Score _{round(accuracy, 2)}_')
    st.dataframe(df_pred, hide_index=True, use_container_width=True)
    st.divider()
