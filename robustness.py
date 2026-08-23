import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve,
)

from data_preprocessing import run_pipeline, RANDOM_STATE

TEST_SIZE = 0.30
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def eval_model(label, pipeline, param_grid, scoring, Xtr, Xte, ytr, yte):
    grid = GridSearchCV(pipeline, param_grid, cv=CV, scoring=scoring, n_jobs=-1, verbose=0)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    y_pred = best.predict(Xte)
    y_prob = best.predict_proba(Xte)[:, 1]
    return {
        "Check": label,
        "Scoring": scoring,
        "Best Params": str(grid.best_params_),
        "Accuracy": round(accuracy_score(yte, y_pred), 4),
        "Precision": round(precision_score(yte, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(yte, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(yte, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(yte, y_prob), 4),
    }, best, y_prob


def best_threshold_from_cv(pipeline, X_train, y_train):
    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    return thresholds[np.nanargmax(f1s[:-1])]


def add_interaction_terms(X):
    X = X.copy()
    X["Age_x_BMI"] = X["Age"] * X["BMI"]
    X["Cholesterol_x_BloodPressure"] = X["Cholesterol Level"] * X["Blood Pressure"]
    X["Sleep_x_Stress"] = X["Sleep Hours"] * X["Stress Level"]
    X["Triglyceride_x_FastingBloodSugar"] = X["Triglyceride Level"] * X["Fasting Blood Sugar"]
    X["CRP_x_Homocysteine"] = X["CRP Level"] * X["Homocysteine Level"]
    return X


DT_GRID_CURRENT = {
    "dt__criterion": ["gini", "entropy"],
    "dt__max_depth": [3, 5, 8],
    "dt__min_samples_leaf": [10, 25, 50],
}
DT_GRID_WIDENED = {
    "dt__criterion": ["gini", "entropy"],
    "dt__max_depth": [3, 5, 8, 12, None],
    "dt__min_samples_leaf": [5, 10, 25, 50],
    "dt__min_samples_split": [2, 10, 20],
    "dt__class_weight": [None, "balanced"],
}
RF_GRID_CURRENT = {
    "rf__n_estimators": [100, 300],
    "rf__min_samples_leaf": [5, 10],
    "rf__class_weight": ["balanced"],
}
RF_GRID_FREE_WEIGHT = {
    "rf__n_estimators": [100, 300],
    "rf__min_samples_leaf": [5, 10],
    "rf__class_weight": [None, "balanced"],
}


if __name__ == "__main__":
    data = run_pipeline("heart_disease.csv")
    X, y = data["X"], data["y"]
    X_eng = add_interaction_terms(X)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    Xtr_eng, Xte_eng, ytr_eng, yte_eng = train_test_split(X_eng, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    rows = []

    for scoring in ["f1", "accuracy", "roc_auc", "recall"]:
        row, _, _ = eval_model(f"DT scoring={scoring}", Pipeline([("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
                                DT_GRID_CURRENT, scoring, Xtr, Xte, ytr, yte)
        rows.append(row)
        row, _, _ = eval_model(f"RF scoring={scoring}", Pipeline([("rf", RandomForestClassifier(random_state=RANDOM_STATE))]),
                                RF_GRID_CURRENT, scoring, Xtr, Xte, ytr, yte)
        rows.append(row)

    row, dt_model, dt_prob = eval_model("DT scoring=f1 (baseline for threshold test)",
                                         Pipeline([("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
                                         DT_GRID_CURRENT, "f1", Xtr, Xte, ytr, yte)
    tuned_t = best_threshold_from_cv(dt_model, Xtr, ytr)
    y_pred_t = (dt_prob >= tuned_t).astype(int)
    rows.append({
        "Check": f"DT threshold-tuned (t={tuned_t:.3f})", "Scoring": "f1 + threshold tuning", "Best Params": "-",
        "Accuracy": round(accuracy_score(yte, y_pred_t), 4), "Precision": round(precision_score(yte, y_pred_t, zero_division=0), 4),
        "Recall": round(recall_score(yte, y_pred_t, zero_division=0), 4), "F1-Score": round(f1_score(yte, y_pred_t, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(yte, dt_prob), 4),
    })

    row, _, _ = eval_model("DT widened grid (class_weight incl.)", Pipeline([("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
                            DT_GRID_WIDENED, "f1", Xtr, Xte, ytr, yte)
    rows.append(row)
    row, _, _ = eval_model("RF class_weight=None allowed", Pipeline([("rf", RandomForestClassifier(random_state=RANDOM_STATE))]),
                            RF_GRID_FREE_WEIGHT, "f1", Xtr, Xte, ytr, yte)
    rows.append(row)

    row, _, _ = eval_model("DT + interaction features", Pipeline([("dt", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
                            DT_GRID_CURRENT, "f1", Xtr_eng, Xte_eng, ytr_eng, yte_eng)
    rows.append(row)
    row, _, _ = eval_model("RF + interaction features", Pipeline([("rf", RandomForestClassifier(random_state=RANDOM_STATE))]),
                            RF_GRID_CURRENT, "f1", Xtr_eng, Xte_eng, ytr_eng, yte_eng)
    rows.append(row)

    df = pd.DataFrame(rows)
    print(df.drop(columns=["Best Params"]).to_string(index=False))