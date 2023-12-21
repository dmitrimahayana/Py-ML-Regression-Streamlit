import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def cleanse_data(df_input: pd.DataFrame):
    # Select specific column
    df_input = df_input[['Unit Weight in grams', 'Underwire', 'Sports', 'Front Opening', 'Size', 'Cup']]

    # Trim value
    df_input.loc[:, 'Underwire'] = df_input['Underwire'].str.strip()
    df_input.loc[:, 'Sports'] = df_input['Sports'].str.strip()
    df_input.loc[:, 'Front Opening'] = df_input['Front Opening'].str.strip()

    # Remove rows where any of the specified columns have empty/null values
    df_input = df_input.dropna(subset=['Unit Weight in grams', 'Size', 'Cup'])

    # Convert columns into integer
    df_input['Unit Weight in grams'] = df_input['Unit Weight in grams'].astype(int)
    df_input['Size'] = df_input['Size'].astype(int)

    # Combine size and cup
    df_input['New Size'] = df_input['Size'].astype(str) + df_input['Cup'].astype(str)

    # Drop size and cup columns
    df_input = df_input.drop('Size', axis=1)
    df_input = df_input.drop('Cup', axis=1)

    # Check Result
    print("\n", df_input.head(5))
    print("\nTotal Row and Col:", df_input.shape)

    return df_input


def encode_data(df_input: pd.DataFrame):
    oe = OrdinalEncoder(encoded_missing_value=-1)
    le = LabelEncoder()
    # ohe = OneHotEncoder(sparse_output=False)

    # Encode column X
    df_input.loc[:, ['Underwire', 'Sports', 'Front Opening']] = oe.fit_transform(
        df_input[['Underwire', 'Sports', 'Front Opening']])
    print("\n", df_input[['Underwire', 'Sports', 'Front Opening']].tail(5))

    # Encode column Y
    # df_input.loc[:, 'New Size'] = le.fit_transform(df_input['New Size'])
    # print("\n", df_input[['New Size']].tail(5))

    # # OHE fit and transform the specified columns
    # encoded_columns = ohe.fit_transform(df_input[['Underwire', 'Sports', 'Front Opening']])
    #
    # # OHE create a DataFrame with the encoded columns
    # encoded_columns_df = pd.DataFrame(encoded_columns,
    #                                   columns=ohe.get_feature_names_out(['Underwire', 'Sports', 'Front Opening']))

    return df_input, oe, le


def split_train_test(df_input: pd.DataFrame):
    # Split data into x (features) and y (target)
    x = df_input.drop('New Size', axis=1)
    y = df_input['New Size'].to_numpy()

    # Scale the data
    scale = StandardScaler()
    x_scaled = scale.fit_transform(x)

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test, scale


if __name__ == "__main__":
    filename = "Glamorise Estimated Weight from Supplier.xlsx"
    # Read input and skip 1st row
    df_input = pd.read_excel(open(f'Dataset/{filename}', 'rb'), skiprows=1,
                             sheet_name='Sheet1')
    print(df_input.dtypes)

    df_input = df_input[['Unit Weight in grams', 'Underwire', 'Sports', 'Front Opening', 'Size', 'Cup']]

    # Trim value
    df_input.loc[:, 'Underwire'] = df_input['Underwire'].str.strip()
    df_input.loc[:, 'Sports'] = df_input['Sports'].str.strip()
    df_input.loc[:, 'Front Opening'] = df_input['Front Opening'].str.strip()

    # Remove rows where any of the specified columns have empty/null values
    df_input = df_input.dropna(subset=['Unit Weight in grams', 'Size', 'Cup'])

    # Convert columns into integer
    df_input['Unit Weight in grams'] = df_input['Unit Weight in grams'].astype(int)
    df_input['Size'] = df_input['Size'].astype(int)

    # Combine size and cup
    df_input['New Size'] = df_input['Size'].astype(str) + df_input['Cup'].astype(str)

    # Drop size and cup columns
    df_input = df_input.drop('Size', axis=1)
    df_input = df_input.drop('Cup', axis=1)

    oe = OrdinalEncoder(encoded_missing_value=-1)
    le = LabelEncoder()

    # Encode column X
    df_input.loc[:, ['Underwire', 'Sports', 'Front Opening']] = oe.fit_transform(
        df_input[['Underwire', 'Sports', 'Front Opening']])

    # Split data into x (features) and y (target)
    x = df_input.drop('New Size', axis=1)
    y = df_input['New Size'].to_numpy()

    # Scale the data
    scale = StandardScaler()
    x_scaled = scale.fit_transform(x)

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    # Predict on test data
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    #
    # # Inverse transform the scaled data numpy format
    # x_original_scaled = scale.inverse_transform(X_test)
    #
    # # Convert this back to a DataFrame (optional, for ease of use)
    # df_x_original_scaled = pd.DataFrame(x_original_scaled, columns=X.columns)
    #
    # # Inverse transform the ordinal encoded columns
    # df_x_original_scaled[['Underwire', 'Sports', 'Front Opening']] = enc.inverse_transform(
    #     df_x_original_scaled[['Underwire', 'Sports', 'Front Opening']]
    # )
    #
    # # Decode the label
    # y_pred_original = le.inverse_transform(y_pred)
    # y_test_original = le.inverse_transform(y_test)
    # print("Some original predictions:", y_pred_original[:5])
    #
    # # Convert Y to df
    # df_y_pred_original = pd.DataFrame(y_pred_original, columns=['New Size Predictions'])
    # df_y_test = pd.DataFrame(y_test_original, columns=['New Size Test'])
    #
    # # Concat Result
    # df_pred = pd.concat([df_x_original_scaled, df_y_pred_original, df_y_test], axis=1)
    #
    # # Save to CSV
    # df_pred.to_excel(f"Output/{filename}", index=False)
