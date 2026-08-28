import os
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

from data_preprocessing import run_pipeline, RANDOM_STATE
from data_visualization import compute_anova_scores

OUTPUT_DIR = "outputs"
TOP_K = 10

sns.set_style("whitegrid")


def run_feature_selection_check(path="heart_disease.csv", top_k=TOP_K, save_outputs=True, data=None):
    """Run the ANOVA top-K feature-selection robustness check (report Section 5.6.5).

    Pass an already-loaded ``data`` dict (as returned by
    ``data_preprocessing.run_pipeline``) to reuse a pipeline result that was
    computed elsewhere (e.g. cached once in a Streamlit app) instead of
    re-running preprocessing from the raw CSV every call.
    """
    if data is None:
        data = run_pipeline(path)
    X, y = data["X"], data["y"]

    anova_results = compute_anova_scores(X, y)
    top_features = anova_results["Feature"].head(top_k).tolist()
    print(f"Top {top_k} features by ANOVA F-score:", top_features)

    X_top = X[top_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_top, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    param_grid = {
        "n_estimators": [100, 300],
        "min_samples_leaf": [5, 10],
        "class_weight": ["balanced"],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": f"RF (Top-{top_k} ANOVA Features)",
        "Best Params": grid.best_params_,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }
    print(metrics)

    # Plot: ANOVA F-scores for top features (for report figure)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    plot_df = anova_results.head(top_k).sort_values("F-Score")
    sns.barplot(x="F-Score", y="Feature", hue="Feature", data=plot_df, palette="viridis", legend=False, ax=ax1)
    ax1.set_title(f"Top {top_k} Features by ANOVA F-Score")
    ax1.set_xlabel("F-Score")
    plt.tight_layout()

    # Plot: confusion matrix for this reduced model
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax2)
    ax2.set_title(f"Confusion Matrix -- RF (Top-{top_k} Features)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    plt.tight_layout()

    # Plot: ROC curve, reduced vs full-feature comparison point
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax3.plot(fpr, tpr, linewidth=2, label=f"Top-{top_k} Features (AUC = {metrics['ROC-AUC']:.3f})")
    ax3.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title(f"ROC Curve -- RF (Top-{top_k} ANOVA Features)")
    ax3.legend(loc="lower right")
    plt.tight_layout()

    if save_outputs:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        anova_results.to_csv(os.path.join(OUTPUT_DIR, "anova_feature_scores.csv"), index=False)
        with open(os.path.join(OUTPUT_DIR, "feature_selection_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        fig1.savefig(os.path.join(OUTPUT_DIR, "feature_selection_anova_scores.png"), dpi=150, bbox_inches="tight")
        fig2.savefig(os.path.join(OUTPUT_DIR, "feature_selection_confusion_matrix.png"), dpi=150, bbox_inches="tight")
        fig3.savefig(os.path.join(OUTPUT_DIR, "feature_selection_roc_curve.png"), dpi=150, bbox_inches="tight")
        print(f"\nSaved metrics + plots to ./{OUTPUT_DIR}/")

    return {
        "top_features": top_features,
        "anova_results": anova_results,
        "metrics": metrics,
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "fig_anova_scores": fig1,
        "fig_confusion_matrix": fig2,
        "fig_roc_curve": fig3,
    }


if __name__ == "__main__":
    run_feature_selection_check()