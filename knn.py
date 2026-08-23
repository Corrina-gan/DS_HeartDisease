import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

from data_preprocessing import run_pipeline, RANDOM_STATE

TEST_SIZE = 0.30
OUTPUT_DIR = "outputs"

KNN_SCORING = "f1"

BASIC_PARAM_GRID = {
    "knn__n_neighbors": [5, 9, 15],
    "knn__metric": ["euclidean", "manhattan"],
}
SMOTE_PARAM_GRID = {
    "knn__n_neighbors": [5, 15, 31],
    "knn__metric": ["euclidean", "manhattan"],
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
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------

def build_basic_pipeline():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])


def build_smote_pipeline():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("knn", KNeighborsClassifier()),
    ])


# --------------------------------------------------------------------------
# Tune + evaluate
# --------------------------------------------------------------------------

def tune_and_evaluate(pipeline, param_grid, X_train, X_test, y_train, y_test, model_name, cv=5, scoring=KNN_SCORING):
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Split": f"{int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}",
        "Best Params": grid.best_params_,
        "CV Scoring Metric": scoring,
        f"CV {scoring.upper()} (best)": round(grid.best_score_, 4),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
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
def get_permutation_importance(model, X_test, y_test):
    """Calculates permutation importance for models without native feature weights."""
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    
    imp_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": result.importances_mean,
        "Std_Dev": result.importances_std
    })
    
    imp_df = imp_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    imp_df.index += 1
    imp_df.index.name = "Rank"
    
    return imp_df.reset_index()

def plot_permutation_importance(imp_df, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    top_df = imp_df.head(15)
    
    sns.barplot(
        x="Importance", 
        y="Feature", 
        data=top_df, 
        palette="viridis", 
        ax=ax
    )
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Mean Accuracy Decrease (Importance)")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig

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
    plt.title(f"ROC Curves: Basic KNN vs SMOTE KNN ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split)")
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
    plt.title("Basic KNN vs SMOTE KNN -- Metric Comparison (70/30 split)")
    plt.legend(title=None, loc="lower right")
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.tight_layout()
    return plt.gcf()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_knn_experiments(path="heart_disease.csv", save_outputs=True, show_plots=True):
    data = run_pipeline(path)
    X, y = data["X"], data["y"]

    X_train, X_test, y_train, y_test = get_70_30_split(X, y)

    section("1. Basic KNN (70/30 split)")
    basic_results = tune_and_evaluate(
        build_basic_pipeline(), BASIC_PARAM_GRID,
        X_train, X_test, y_train, y_test, "1. Basic KNN",
        scoring=KNN_SCORING,
    )
    print(f"Best Params: {basic_results['metrics']['Best Params']}")
    print(f"Best CV {KNN_SCORING.upper()}: {basic_results['metrics'][f'CV {KNN_SCORING.upper()} (best)']}")
    print(basic_results["classification_report"])

    section("2. SMOTE KNN (70/30 split)")
    smote_results = tune_and_evaluate(
        build_smote_pipeline(), SMOTE_PARAM_GRID,
        X_train, X_test, y_train, y_test, "2. SMOTE KNN",
        scoring=KNN_SCORING,
    )
    print(f"Best Params: {smote_results['metrics']['Best Params']}")
    print(f"Best CV {KNN_SCORING.upper()}: {smote_results['metrics'][f'CV {KNN_SCORING.upper()} (best)']}")
    print(smote_results["classification_report"])

    all_results = {"1. Basic KNN": basic_results, "2. SMOTE KNN": smote_results}
    results_df = pd.DataFrame([basic_results["metrics"], smote_results["metrics"]])

    section("Comparison: Basic KNN vs SMOTE KNN (70/30 split)")
    print(results_df.drop(columns=["Best Params"]).to_string(index=False))

    fig_cm = plot_confusion_matrices(all_results)
    fig_roc = plot_roc_curves(all_results, y_test)
    fig_metrics = plot_metric_comparison(results_df)
    if show_plots:
        plt.show()

    if save_outputs:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        results_df.to_csv(os.path.join(OUTPUT_DIR, "knn_metrics_comparison.csv"), index=False)

        json_ready = {
            name: {
                "metrics": res["metrics"],
                "classification_report": res["classification_report"],
                "confusion_matrix": res["confusion_matrix"].tolist(),
            }
            for name, res in all_results.items()
        }
        with open(os.path.join(OUTPUT_DIR, "knn_metrics.json"), "w") as f:
            json.dump(json_ready, f, indent=2)

        fig_cm.savefig(os.path.join(OUTPUT_DIR, "knn_confusion_matrices.png"), dpi=150, bbox_inches="tight")
        fig_roc.savefig(os.path.join(OUTPUT_DIR, "knn_roc_curves.png"), dpi=150, bbox_inches="tight")
        fig_metrics.savefig(os.path.join(OUTPUT_DIR, "knn_metric_comparison.png"), dpi=150, bbox_inches="tight")
        print(f"\nSaved metrics + plots to ./{OUTPUT_DIR}/")

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "results": all_results,
        "results_df": results_df,
    }


if __name__ == "__main__":
    run_knn_experiments()