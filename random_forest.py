
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

from data_preprocessing import run_pipeline, RANDOM_STATE

# Configuration
TEST_SIZE = 0.30
OUTPUT_DIR = "outputs"

# Use F1-score to select the best model during tuning
BASIC_SCORING = "f1"
SMOTE_SCORING = "f1"

# Parameters tested for the basic Random Forest
BASIC_PARAM_GRID = {
    "rf__n_estimators": [100, 300],
    "rf__max_depth": [None],
    "rf__min_samples_leaf": [5, 10],
    "rf__class_weight": [None, "balanced", "balanced_subsample"],
}

# Parameters tested for the SMOTE Random Forest
SMOTE_PARAM_GRID = {
    "rf__n_estimators": [100, 300],
    "rf__max_depth": [5, 8, None],
    "rf__min_samples_leaf": [1, 5, 10],
}

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7, 5)

def section(title):
    """Print a formatted section heading."""
    width = 70
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)

# Train-Test Split
def get_70_30_split(X, y, random_state=RANDOM_STATE):
    """Stratified 70/30 train-test split, matching the split used by knn.py so
    the two models are directly comparable."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )

# Model Pipelines
# Create the standard Random Forest pipeline.
def build_basic_pipeline():
    return ImbPipeline([
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])

# Create a pipeline with scaling, SMOTE, and Random Forest
# Scaling is needed because SMOTE uses distance between samples
def build_smote_pipeline():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])

# Model Tuning and Evaluation
def tune_and_evaluate(pipeline, param_grid, X_train, X_test, y_train, y_test, model_name, cv=5, scoring=SMOTE_SCORING):
    # Find the best parameter combination using cross-validation
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

    # Make predictions on unseen test data
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
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


# Feature Importance
def get_permutation_importance(model, X_test, y_test):
    # Shuffle each feature and observe how much F1-score decreases
    result = permutation_importance(model, X_test, y_test, scoring="f1", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)

    imp_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": result.importances_mean,
        "Std_Dev": result.importances_std
    })

    # Rank features from most to least important
    imp_df = imp_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    imp_df.index += 1
    imp_df.index.name = "Rank"

    return imp_df.reset_index()

def plot_permutation_importance(imp_df, title, top_n=None):
    """Plot permutation importance for the selected features."""

    # Display all features or only the top N features
    plot_df = imp_df if top_n is None else imp_df.head(top_n)

    # Increase plot height when there are many features
    height = max(6, 0.32 * len(plot_df))
    fig, ax = plt.subplots(figsize=(8, height))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=plot_df,
        palette="viridis",
        ax=ax
    )

    ax.axvline(0, color="grey", linewidth=1)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Mean Accuracy Decrease (Importance)")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig

# Confusion Matrix Plot
def plot_confusion_matrices(all_results, class_labels=("No", "Yes")):
    """Plot confusion matrices for all models."""
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        # Display actual vs predicted classifications
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

# ROC Curve Plot
def plot_roc_curves(all_results, y_test):
    """Plot ROC curves and AUC values for all models."""
    plt.figure(figsize=(7, 6))
    for name, res in all_results.items():
        # Calculate false positive and true positive rates
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        auc = res["metrics"]["ROC-AUC"]

        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")

    # Reference line representing random classification
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: Basic RF vs SMOTE RF ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split)")

    plt.legend(loc="lower right")
    plt.tight_layout()

    return plt.gcf()

# Metric Comparison Plot
def plot_metric_comparison(results_df):
    """Compare evaluation metrics between the two Random Forest models."""

    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    # Convert results into a format suitable for plotting
    plot_df = results_df.melt(id_vars="Model", value_vars=metric_cols,
                               var_name="Metric", value_name="Score")
    plt.figure(figsize=(9, 5.5))

    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="Set2")

    plt.ylim(0, 1)

    plt.title("Basic Random Forest vs SMOTE Random Forest -- Metric Comparison (70/30 split)")

    plt.legend(title=None, loc="lower right")

    # Display metric values above each bar
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.tight_layout()
    return plt.gcf()

# Main Experiment
def run_random_forest_experiments(path="heart_disease.csv", save_outputs=True, show_plots=True):
    # Load and preprocess the dataset
    data = run_pipeline(path)
    X, y = data["X"], data["y"]

    # Create the same 70/30 split for both models
    X_train, X_test, y_train, y_test = get_70_30_split(X, y)

    section("1. Basic Random Forest (70/30 split)")

    # 1. Basic Random Forest
    basic_results = tune_and_evaluate(
        build_basic_pipeline(), BASIC_PARAM_GRID,
        X_train, X_test, y_train, y_test, "1. Basic Random Forest",
        scoring=BASIC_SCORING,
    )

    print(f"Best Params: {basic_results['metrics']['Best Params']}")
    print(f"Best CV {BASIC_SCORING.upper()}: {basic_results['metrics'][f'CV {BASIC_SCORING.upper()} (best)']}")
    print(basic_results["classification_report"])

    # 2. SMOTE Random Forest
    section("2. SMOTE Random Forest (70/30 split)")

    smote_results = tune_and_evaluate(
        build_smote_pipeline(), SMOTE_PARAM_GRID,
        X_train, X_test, y_train, y_test, "2. SMOTE Random Forest",
        scoring=SMOTE_SCORING,
    )

    print(f"Best Params: {smote_results['metrics']['Best Params']}")
    print(f"Best CV {SMOTE_SCORING.upper()}: {smote_results['metrics'][f'CV {SMOTE_SCORING.upper()} (best)']}")
    print(smote_results["classification_report"])

    # Store results for both models
    all_results = {
        "1. Basic Random Forest": basic_results,
        "2. SMOTE Random Forest": smote_results,
    }

    results_df = pd.DataFrame([basic_results["metrics"], smote_results["metrics"]])

    # Compare Models
    section("Comparison: Basic RF vs SMOTE RF (70/30 split)")
    print(results_df.drop(columns=["Best Params"]).to_string(index=False))

    # Create comparison visualizations
    fig_cm = plot_confusion_matrices(all_results)
    fig_roc = plot_roc_curves(all_results, y_test)
    fig_metrics = plot_metric_comparison(results_df)

    if show_plots:
        plt.show()

    # Save Results
    if save_outputs:
        # Create output folder if it does not exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save metrics as CSV
        results_df.to_csv(os.path.join(OUTPUT_DIR, "random_forest_metrics_comparison.csv"), index=False)

        # Prepare results for JSON format
        json_ready = {
            name: {
                "metrics": res["metrics"],
                "classification_report": res["classification_report"],
                "confusion_matrix": res["confusion_matrix"].tolist(),
            }
            for name, res in all_results.items()
        }
         # Save metrics and evaluation results
        with open(os.path.join(OUTPUT_DIR, "random_forest_metrics.json"), "w") as f:
            json.dump(json_ready, f, indent=2)

        # Save generated plots
        fig_cm.savefig(os.path.join(OUTPUT_DIR, "random_forest_confusion_matrices.png"), dpi=150, bbox_inches="tight")
        fig_roc.savefig(os.path.join(OUTPUT_DIR, "random_forest_roc_curves.png"), dpi=150, bbox_inches="tight")
        fig_metrics.savefig(os.path.join(OUTPUT_DIR, "random_forest_metric_comparison.png"), dpi=150, bbox_inches="tight")

        print(f"\nSaved metrics + plots to ./{OUTPUT_DIR}/")
    # Return all data and model results for further analysis
    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "results": all_results,
        "results_df": results_df,
    }

# Run the experiment when this file is executed 
if __name__ == "__main__":
    run_random_forest_experiments()
