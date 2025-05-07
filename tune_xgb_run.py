from src.tune_xgb import optimize_xgboost
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import joblib

# Load preprocessed features
X_train = joblib.load("artifacts/X_train.pkl")
X_test = joblib.load("artifacts/X_test.pkl")
y_train = joblib.load("artifacts/y_train.pkl")
y_test = joblib.load("artifacts/y_test.pkl")

# Step 1: Tune
print("🔍 Tuning XGBoost with Optuna...")
best_params = optimize_xgboost(X_train, y_train)

# Add fixed params
best_params['use_label_encoder'] = False
best_params['eval_metric'] = 'logloss'

# Step 2: Train with best params
print("✅ Training model...")
model = XGBClassifier(**best_params)
model.fit(X_train, y_train)

# Step 3: Evaluate
y_pred = model.predict(X_test)
print("\n📊 Tuned XGBoost Metrics:")
print(f"F1 Score:      {f1_score(y_test, y_pred):.4f}")
print(f"Precision:     {precision_score(y_test, y_pred):.4f}")
print(f"Recall:        {recall_score(y_test, y_pred):.4f}")
