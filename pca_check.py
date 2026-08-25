import os
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

from data_preprocessing import run_pipeline, RANDOM_STATE

OUTPUT_DIR = "outputs"
VARIANCE_RETAINED = 0.95

sns.set_style("whitegrid")


def run_pca_check(path="heart_disease.csv", variance_retained=VARIANCE_RETAINED, save_outputs=True):
    data = run_pipeline(path)
    X, y = data["X"], data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=variance_retained, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    n_original = X.shape[1]
    n_components = pca.n_components_
    print(f"Original features: {n_original}")
    print(f"Components needed for {variance_retained*100:.0f}% variance: {n_components}")

    param_grid = {
        "n_estimators": [100, 300],
        "min_samples_leaf": [5, 10],
        "class_weight": ["balanced"],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1,
    )
    grid.fit(X_train_pca, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test_pca)
    y_prob = best_model.predict_proba(X_test_pca)[:, 1]

    metrics = {
        "Model": f"RF (PCA, {n_components} components, {variance_retained*100:.0f}% variance)",
        "Original Features": n_original,
        "PCA Components": n_components,
        "Best Params": grid.best_params_,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }
    print(metrics)

    # Plot 1: cumulative explained variance (scree plot)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    pca_full = PCA(random_state=RANDOM_STATE).fit(X_train_s)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    ax1.plot(range(1, len(cumvar) + 1), cumvar, marker="o", markersize=3)
    ax1.axhline(variance_retained, color="red", linestyle="--", label=f"{variance_retained*100:.0f}% variance")
    ax1.axvline(n_components, color="grey", linestyle="--", label=f"{n_components} components")
    ax1.set_xlabel("Number of Principal Components")
    ax1.set_ylabel("Cumulative Explained Variance")
    ax1.set_title("PCA -- Cumulative Explained Variance")
    ax1.legend(loc="lower right")
    plt.tight_layout()

    # Plot 2: confusion matrix
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax2)
    ax2.set_title(f"Confusion Matrix -- RF (PCA, {n_components} components)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    plt.tight_layout()

    # Plot 3: ROC curve
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax3.plot(fpr, tpr, linewidth=2, label=f"PCA RF (AUC = {metrics['ROC-AUC']:.3f})")
    ax3.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve -- RF on PCA-Reduced Features")
    ax3.legend(loc="lower right")
    plt.tight_layout()

    if save_outputs:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "pca_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        fig1.savefig(os.path.join(OUTPUT_DIR, "pca_explained_variance.png"), dpi=150, bbox_inches="tight")
        fig2.savefig(os.path.join(OUTPUT_DIR, "pca_confusion_matrix.png"), dpi=150, bbox_inches="tight")
        fig3.savefig(os.path.join(OUTPUT_DIR, "pca_roc_curve.png"), dpi=150, bbox_inches="tight")
        print(f"\nSaved metrics + plots to ./{OUTPUT_DIR}/")

    return {
        "metrics": metrics,
        "n_components": n_components,
        "pca": pca,
        "best_model": best_model,
    }


if __name__ == "__main__":
    run_pca_check()