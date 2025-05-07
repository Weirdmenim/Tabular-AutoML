import joblib
import os
import json
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier, plot_importance
from src.model_selector import evaluate_models
from src.evaluate import evaluate_on_test
import numpy as np

# Directory for artifacts
ARTIFACTS_DIR = "artifacts"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load preprocessed data
try:
    X_train = joblib.load(os.path.join(ARTIFACTS_DIR, "X_train.pkl"))
    X_test = joblib.load(os.path.join(ARTIFACTS_DIR, "X_test.pkl"))
    y_train = joblib.load(os.path.join(ARTIFACTS_DIR, "y_train.pkl"))
    y_test = joblib.load(os.path.join(ARTIFACTS_DIR, "y_test.pkl"))
    print("Preprocessing complete.")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Load feature names once
feature_names_path = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")
try:
    feature_names = joblib.load(feature_names_path)
    print("Feature names loaded successfully.")
except Exception as e:
    print(f"Error loading feature names: {e}")
    feature_names = None

# Check for tuned XGBoost parameters
best_model = None
best_model_name = ""
best_xgb_params_path = os.path.join(ARTIFACTS_DIR, "best_xgb_params.json")

if os.path.exists(best_xgb_params_path):
    print("⚙️ Loading tuned XGBoost parameters...")
    try:
        with open(best_xgb_params_path, "r") as f:
            best_xgb_params = json.load(f)
        best_xgb_params.pop('use_label_encoder', None)
        best_model = XGBClassifier(**best_xgb_params)
        best_model.fit(X_train, y_train)
        best_model_name = "Tuned XGBoost"
    except Exception as e:
        print(f"Error loading or fitting tuned XGBoost: {e}")
        raise
else:
    print("🚀 Running model selection (no tuned parameters found)...")
    try:
        best_model_name, best_model = evaluate_models(X_train, y_train)
        if best_model is None:
            print("❌ No models trained successfully.")
            raise ValueError("Model selection failed to return a valid model.")
    except Exception as e:
        print(f"Error during model selection: {e}")
        raise

# Evaluate the best model on the test set
if best_model is not None:
    print(f"\n✅ Best model: {best_model_name}")
    try:
        evaluate_on_test(best_model, X_test, y_test)
    except Exception as e:
        print(f"Error evaluating model: {e}")
        raise
else:
    print("No model was successfully trained. Skipping evaluation.")

# Feature importance and SHAP for XGBoost
if isinstance(best_model, XGBClassifier) and feature_names is not None:
    print("\n🔍 Interpreting XGBoost model with feature importance and SHAP...")

    # Plot XGBoost feature importance for different types
    for imp_type in ["weight", "gain", "cover"]:
        plot_importance(best_model, importance_type=imp_type, max_num_features=15)
        plt.title(f"XGBoost Feature Importance ({imp_type})")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"xgb_feature_importance_{imp_type}.png"))
        plt.close()

    # Plot top 20 feature importances as a bar chart
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.title("Top 20 Feature Importances")
    plt.barh(top_features[::-1], top_importances[::-1])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "top_20_feature_importances.png"))
    plt.close()

    # SHAP interpretation
    explainer = shap.Explainer(best_model, X_train, feature_names=feature_names)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_summary_plot.png"))
    plt.close()

    print("📊 Feature importance plots and SHAP summary saved to results/")
else:
    if not isinstance(best_model, XGBClassifier):
        print("\nℹ️ Feature importance and SHAP are only implemented for XGBoost models.")
    if feature_names is None:
        print("\n⚠️ Feature names not available. Skipping feature importance and SHAP.")