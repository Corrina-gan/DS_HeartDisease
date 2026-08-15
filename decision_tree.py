
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from data_preprocessing import run_pipeline, RANDOM_STATE

TEST_SIZE = 0.30
OUTPUT_DIR = "outputs"

BASIC_PARAM_GRID = {
    "dt__criterion": ["gini", "entropy"],
    "dt__max_depth": [3, 5, 8],
    "dt__min_samples_leaf": [10, 25, 50],
}
SMOTE_PARAM_GRID = {
    "dt__criterion": ["gini", "entropy"],
    "dt__max_depth": [3, 5, 8],
    "dt__min_samples_leaf": [10, 25, 50],
}

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7, 5)


def section(title):
    width = 70
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

def get_70_30_split(X, y, random_state=RANDOM_STATE):
    """Stratified 70/30 train-test split, matching the split used by knn.py so
    the two models are directly comparable."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------

def build_basic_pipeline():
    return ImbPipeline([
        ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])


def build_smote_pipeline():
    # SMOTE interpolates using Euclidean distance, so features are scaled first
    # even though the tree itself is scale-invariant.
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])


# --------------------------------------------------------------------------
# Tune + evaluate
# --------------------------------------------------------------------------

def tune_and_evaluate(pipeline, param_grid, X_train, X_test, y_train, y_test, model_name, cv=5):
    grid = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Split": f"{int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}",
        "Best Params": grid.best_params_,
        "CV ROC-AUC (best)": round(grid.best_score_, 4),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }

    return {
        "grid": grid,
        "best_model": best_model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


# --------------------------------------------------------------------------
# Visualizations
# --------------------------------------------------------------------------

def plot_confusion_matrices(all_results, class_labels=("No", "Yes")):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, all_results.items()):
        sns.heatmap(
            res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels, cbar=False, ax=ax,
        )
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle(f"Confusion Matrices ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split)", fontsize=14)
    plt.tight_layout()
    return fig


def plot_roc_curves(all_results, y_test):
    plt.figure(figsize=(7, 6))
    for name, res in all_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        auc = res["metrics"]["ROC-AUC"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: Basic DT vs SMOTE DT ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt.gcf()


def plot_metric_comparison(results_df):
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    plot_df = results_df.melt(id_vars="Model", value_vars=metric_cols,
                               var_name="Metric", value_name="Score")
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="Set2")
    plt.ylim(0, 1)
    plt.title("Basic Decision Tree vs SMOTE Decision Tree -- Metric Comparison (70/30 split)")
    plt.legend(title=None, loc="lower right")
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.tight_layout()
    return plt.gcf()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_decision_tree_experiments(path="heart_disease.csv", save_outputs=True, show_plots=True):
    data = run_pipeline(path)
    X, y = data["X"], data["y"]

    X_train, X_test, y_train, y_test = get_70_30_split(X, y)

    section("1. Basic Decision Tree (70/30 split)")
    basic_results = tune_and_evaluate(
        build_basic_pipeline(), BASIC_PARAM_GRID,
        X_train, X_test, y_train, y_test, "1. Basic Decision Tree",
    )
    print(f"Best Params: {basic_results['metrics']['Best Params']}")
    print(f"Best CV ROC-AUC: {basic_results['metrics']['CV ROC-AUC (best)']}")
    print(basic_results["classification_report"])

    section("2. SMOTE Decision Tree (70/30 split)")
    smote_results = tune_and_evaluate(
        build_smote_pipeline(), SMOTE_PARAM_GRID,
        X_train, X_test, y_train, y_test, "2. SMOTE Decision Tree",
    )
    print(f"Best Params: {smote_results['metrics']['Best Params']}")
    print(f"Best CV ROC-AUC: {smote_results['metrics']['CV ROC-AUC (best)']}")
    print(smote_results["classification_report"])

    all_results = {
        "1. Basic Decision Tree": basic_results,
        "2. SMOTE Decision Tree": smote_results,
    }
    results_df = pd.DataFrame([basic_results["metrics"], smote_results["metrics"]])

    section("Comparison: Basic DT vs SMOTE DT (70/30 split)")
    print(results_df.drop(columns=["Best Params"]).to_string(index=False))

    fig_cm = plot_confusion_matrices(all_results)
    fig_roc = plot_roc_curves(all_results, y_test)
    fig_metrics = plot_metric_comparison(results_df)
    if show_plots:
        plt.show()

    if save_outputs:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        results_df.to_csv(os.path.join(OUTPUT_DIR, "decision_tree_metrics_comparison.csv"), index=False)

        json_ready = {
            name: {
                "metrics": res["metrics"],
                "classification_report": res["classification_report"],
                "confusion_matrix": res["confusion_matrix"].tolist(),
            }
            for name, res in all_results.items()
        }
        with open(os.path.join(OUTPUT_DIR, "decision_tree_metrics.json"), "w") as f:
            json.dump(json_ready, f, indent=2)

        fig_cm.savefig(os.path.join(OUTPUT_DIR, "decision_tree_confusion_matrices.png"), dpi=150, bbox_inches="tight")
        fig_roc.savefig(os.path.join(OUTPUT_DIR, "decision_tree_roc_curves.png"), dpi=150, bbox_inches="tight")
        fig_metrics.savefig(os.path.join(OUTPUT_DIR, "decision_tree_metric_comparison.png"), dpi=150, bbox_inches="tight")
        print(f"\nSaved metrics + plots to ./{OUTPUT_DIR}/")

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "results": all_results,
        "results_df": results_df,
    }


if __name__ == "__main__":
    run_decision_tree_experiments()
