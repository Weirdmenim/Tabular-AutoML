import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

def optimize_xgboost(X, y):
    def objective(trial):
        clf = XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 300),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            gamma=trial.suggest_float('gamma', 0, 5),
            use_label_encoder=False,
            eval_metric='logloss'
        )
        return cross_val_score(clf, X, y, scoring=make_scorer(f1_score), cv=3).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    return study.best_params
