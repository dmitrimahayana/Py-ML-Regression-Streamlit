import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load


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

    # Check Result
    print("\n", df_input.head(5))
    print("\nTotal Row and Col:", df_input.shape)

    return df_input


if __name__ == "__main__":
    # Define dataset
    filename = "Glamorise Estimated Weight from Supplier.xlsx"
    file_path = f"Dataset/{filename}"

    # Read input and skip 1st row
    df_input = pd.read_excel(open(file_path, 'rb'), skiprows=1, sheet_name='Sheet1')
    print(df_input.dtypes)

    # Data cleansing
    df_input = cleanse_data(df_input)

    # Define the columns to be used in each transformer
    categorical_features = ['Underwire', 'Sports', 'Front Opening']
    target_feature = 'New Size'

    # Create a ColumnTransformer to handle different feature types
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(encoded_missing_value=-1), categorical_features),
        ], remainder='passthrough')  # drop or passthrough

    # List of pipelines
    verbose = 0
    pipeline_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_scaling', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, verbose=verbose))
    ])
    pipeline_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_scaling', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, verbose=verbose))
    ])
    pipeline_dt = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_scaling', StandardScaler()),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    pipeline_svm = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_scaling', StandardScaler()),
        ('classifier', SVC(random_state=42, verbose=verbose))
    ])
    pipeline_knn = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_scaling', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ])

    # List of models to evaluate
    pipelines = {
        'LogisticRegression': pipeline_lr,
        'DecisionTree': pipeline_dt,
        'RandomForest': pipeline_rf,
        'SVM': pipeline_svm,
        'KNN': pipeline_knn
    }

    # Define hyperparameters for each pipeline
    hyperparameters = {
        'LogisticRegression': {'classifier__C': [0.1, 1, 10]},
        'DecisionTree': {'classifier__max_depth': [10, 15, 20], 'classifier__min_samples_split': [2, 4, 6]},
        'RandomForest': {'classifier__n_estimators': [10, 50, 100], 'classifier__max_depth': [10, 20, 30]},
        'SVM': {'classifier__C': [0.1, 1, 10], 'classifier__gamma': [1, 0.1, 0.01]},
        'KNN': {'classifier__n_neighbors': [3, 5, 7], 'classifier__weights': ['uniform', 'distance']}
    }

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(df_input.drop(target_feature, axis=1), df_input[target_feature],
                                                        test_size=0.2, random_state=42)

    # Iteration to find best estimator
    best_models = {}
    for model_name, pipe in pipelines.items():
        # Perform grid search
        grid_search = GridSearchCV(pipe, hyperparameters[model_name], cv=5, verbose=verbose, n_jobs=-1)

        # Train the model inside Grid>Pipeline
        grid_search.fit(x_train, y_train)

        # Add model and params
        best_models[model_name] = {
            'name': model_name,
            'model': grid_search.best_estimator_,
            'params': grid_search.best_params_
        }

    for model_name, model_info in best_models.items():
        # Check Accuracy
        accuracy = model_info['model'].score(x_test, y_test)
        print(f"\n1st Accuracy {model_name}: {accuracy}")
        y_pred = model_info['model'].predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"2nd Accuracy {model_name}: {accuracy}")

        # Save model
        model_folder = f"Model/Glamorise Estimated Weight/"
        model_filename = f"{model_info['name']}.joblib"
        isExist = os.path.exists(model_folder)
        if not isExist:
            os.makedirs(model_folder)
            print(f"The new directory {model_folder} is created!")
        # Create a new model file
        dump(model_info['model'], model_folder + model_filename)
        print(f"The best model is saved as {model_folder + model_filename}")

    # Load model from local folder
    model_folder = f"Model/Glamorise Estimated Weight/"
    model_name = "KNN"
    model_filename = f"{model_name}.joblib"
    loaded_model = load(model_folder + model_filename)

    # Predict data
    y_pred = loaded_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nLoad Model and Get Accuracy {model_name}: {accuracy}")

    # Convert prediction into dataframe
    df_y_pred = pd.DataFrame(y_pred, columns=['Predictions'])

    # Reset index on x_test to align with y_pred_df if needed
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Concatenate column-wise
    df_combined = pd.concat([x_test, y_test, df_y_pred], axis=1)

    # Save to CSV
    df_combined.to_excel(f"Output/{filename}", index=False)
    print(f"result saved to Output/{filename}")
