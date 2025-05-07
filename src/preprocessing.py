import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set paths
DATA_PATH = os.path.join("data", "adults.csv")
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_data():
    """Load the Adult Income dataset from CSV."""
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income"
    ]
    df = pd.read_csv(DATA_PATH, header=None, names=column_names, na_values=" ?", skipinitialspace=True)
    return df

def preprocess_data(df):
    """Preprocess the dataset and return processed features, labels, and feature names."""
    # Drop rows with missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop("income", axis=1)
    y = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Define preprocessing pipelines
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine pipelines in a ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # Extract feature names after transformation
    cat_encoder = preprocessor.named_transformers_["cat"]["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_feature_names)

    # Save preprocessor and feature names
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))
    joblib.dump(feature_names, os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))

    return X_processed, y, feature_names

def main():
    """Main function to load, preprocess, split, and save the dataset."""
    # Load data
    df = load_data()
    print("Dataset Loaded.")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Preprocess data
    X_processed, y, feature_names = preprocess_data(df)
    print(f"Processed data shape: {X_processed.shape}")
    print(f"Number of features: {len(feature_names)}")

    # Split into train and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save processed datasets
    joblib.dump(X_train, os.path.join(ARTIFACTS_DIR, "X_train.pkl"))
    joblib.dump(X_test, os.path.join(ARTIFACTS_DIR, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(ARTIFACTS_DIR, "y_train.pkl"))
    joblib.dump(y_test, os.path.join(ARTIFACTS_DIR, "y_test.pkl"))

    print("Preprocessing complete.")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

if __name__ == "__main__":
    main()