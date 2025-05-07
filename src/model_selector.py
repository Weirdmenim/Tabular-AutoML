from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import joblib
import numpy as np

def evaluate_models(X_train, y_train):
    """
    Evaluates several classification models using cross-validation,
    selects the best one based on F1 score, fits it to the full
    training data, and saves it.

    Args:
        X_train (np.ndarray or pd.DataFrame): Training features.
        y_train (np.ndarray or pd.Series): Training labels.

    Returns:
        tuple: A tuple containing:
            - str: The name of the best performing model.
            - model object: The fitted best performing model instance.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        # CORRECTED LINE: Removed the 'model =' assignment
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        # Use try-except for potential errors during cross-validation (e.g., convergence)
        try:
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            mean_score = np.mean(scores)
            results[name] = {
                "model": model,
                "cv_f1": mean_score
            }
            print(f"{name} F1 (cv): {mean_score:.4f}")
        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = {
                "model": model,
                "cv_f1": -1 # Assign a low score on error so it's not chosen as best
            }


    # Pick best
    # Ensure there is at least one valid result before finding the max
    if not results or all(r['cv_f1'] == -1 for r in results.values()):
        print("❌ No models trained successfully.")
        return None, None

    best_model_name = max(results, key=lambda k: results[k]['cv_f1'])
    # Retrieve the original model instance from the dictionary entry
    best_model = results[best_model_name]['model']

    print(f"\n✅ Best model: {best_model_name} with CV F1 = {results[best_model_name]['cv_f1']:.4f}")

    # Fit on all training data
    print(f"Fitting {best_model_name} on full training data...")
    try:
        best_model.fit(X_train, y_train)
        print("Fit successful.")
    except Exception as e:
         print(f"Error fitting best model ({best_model_name}) on full training data: {e}")
         return None, None # Indicate failure


    # Save model
    # Ensure the 'results' directory exists
    import os
    if not os.path.exists("results"):
        os.makedirs("results")
        print("Created 'results' directory.")

    save_path = "results/best_model.joblib"
    try:
        joblib.dump(best_model, save_path)
        print(f"✅ Best model saved to {save_path}")
    except Exception as e:
        print(f"Error saving model to {save_path}: {e}")
        # Decide whether to return the model even if saving fails
        # For now, let's return None to indicate a partial failure
        return None, None # Or return best_model_name, best_model if you want to return even on save failure


    return best_model_name, best_model

# Example Usage (assuming X_train and y_train are defined elsewhere)
# if __name__ == "__main__":
#     # Load your data here to define X_train and y_train
#     # from sklearn.model_selection import train_test_split
#     # X, y = ... load your data ...
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Replace with actual X_train, y_train
#     # For testing purposes only:
#     X_train_dummy = np.random.rand(100, 10)
#     y_train_dummy = np.random.randint(0, 2, 100)
#
#     best_name, fitted_best_model = evaluate_models(X_train_dummy, y_train_dummy)
#
#     if fitted_best_model:
#         print(f"\nEvaluation complete. Best model found: {best_name}")
#         # You can now use fitted_best_model for predictions
#         # predictions = fitted_best_model.predict(X_test) # Assuming X_test is available