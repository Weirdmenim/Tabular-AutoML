import pandas as pd
from src.preprocessing import preprocess_data
from src.model_selector import evaluate_models
from src.evaluate import evaluate_on_test

def load_data():
    path = "data/adults.csv"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    df = pd.read_csv(path, names=columns, skiprows=1)
    print("Dataset Loaded.")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    print("Preprocessing complete.")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

best_model_name, best_model = evaluate_models(X_train, y_train)

evaluate_on_test(best_model, X_test, y_test)