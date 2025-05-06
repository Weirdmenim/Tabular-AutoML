from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import joblib
import numpy as np

def evaluate_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        mean_score = np.mean(scores)
        results[name] = {
            "model": model,
            "cv_f1": mean_score
        }
        print(f"{name} F1 (cv): {mean_score:.4f}")
    
    # Pick best
    best_model_name = max(results, key=lambda k: results[k]['cv_f1'])
    best_model = results[best_model_name]['model']
    print(f"\n✅ Best model: {best_model_name}")

    # Fit on all training data
    best_model.fit(X_train, y_train)

    # Save model
    joblib.dump(best_model, "results/best_model.joblib")

    return best_model_name, best_model
