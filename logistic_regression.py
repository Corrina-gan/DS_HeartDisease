
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
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

# newton-cg / sag / newton-cholesky only support L2, so only liblinear, lbfgs
# and saga are searched -- liblinear/saga also support L1 (automatic feature
# selection) while lbfgs is L2-only but tends to converge fastest.
_C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]
_CLASS_WEIGHTS = [None, "balanced"]


def _logreg_param_grid(prefix="logreg"):
    return [
        {
            f"{prefix}__solver": ["liblinear"],
            f"{prefix}__penalty": ["l1", "l2"],
            f"{prefix}__C": _C_VALUES,
            f"{prefix}__class_weight": _CLASS_WEIGHTS,
        },
        {
            f"{prefix}__solver": ["lbfgs"],
            f"{prefix}__penalty": ["l2"],
            f"{prefix}__C": _C_VALUES,
            f"{prefix}__class_weight": _CLASS_WEIGHTS,
        },
        {
            f"{prefix}__solver": ["saga"],
            f"{prefix}__penalty": ["l1", "l2"],
            f"{prefix}__C": _C_VALUES,
            f"{prefix}__class_weight": _CLASS_WEIGHTS,
        },
    ]


BASIC_PARAM_GRID = _logreg_param_grid("logreg")
SMOTE_PARAM_GRID = _logreg_param_grid("logreg")

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
    """Stratified 70/30 train-test split (independent of the 80/20 split
    produced by data_preprocessing.split_and_scale, which other models may
    still use)."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------

def build_basic_pipeline():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(random_state=RANDOM_STATE, max_iter=5000)),
    ])


def build_smote_pipeline():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("logreg", LogisticRegression(random_state=RANDOM_STATE, max_iter=5000)),
    ])


# --------------------------------------------------------------------------
# Tune + evaluate
# --------------------------------------------------------------------------

def tune_and_evaluate(pipeline, param_grid, X_train, X_test, y_train, y_test, model_name, cv=5):
    # Scored on F1 (not ROC-AUC) so the search picks hyperparameters -- and a
    # class_weight -- that actually detect the positive class at the default
    # 0.5 threshold, instead of hyperparameters that only rank well.
    # n_jobs capped (not -1): this grid has 60 candidates x 5 folds, and on a
    # high-core/low-RAM machine "one worker per core" can spawn enough loky
    # processes to exhaust memory mid-search (observed: MemoryError /
    # BrokenProcessPool while pickling tasks to workers).
    n_jobs = min(4, os.cpu_count() or 1)
    grid = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="f1", n_jobs=n_jobs, verbose=1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Split": f"{int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}",
        "Best Params": grid.best_params_,
        "CV F1 (best)": round(grid.best_score_, 4),
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
# Coefficients (Logistic Regression interpretability)
# --------------------------------------------------------------------------

def get_coefficients(best_model, feature_names, step_name="logreg"):
    coefs = best_model.named_steps[step_name].coef_[0]

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
    })
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
    coef_df.insert(0, "Rank", range(1, len(coef_df) + 1))
    coef_df["Coefficient"] = coef_df["Coefficient"].round(4)
    coef_df["Abs_Coefficient"] = coef_df["Abs_Coefficient"].round(4)
    coef_df["Effect"] = np.where(
        coef_df["Coefficient"] > 0, "Increases risk", "Decreases risk"
    )
    return coef_df


def plot_coefficients(coef_df, title="Logistic Regression Coefficients (Feature Influence)"):
    plot_df = coef_df.sort_values("Coefficient")
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in plot_df["Coefficient"]]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["Feature"], plot_df["Coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Coefficient (negative = lowers risk, positive = raises risk)")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig


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
    plt.title(f"ROC Curves: Basic vs SMOTE Logistic Regression ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split)")
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
    plt.title("Basic vs SMOTE Logistic Regression -- Metric Comparison (70/30 split)")
    plt.legend(title=None, loc="lower right")
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.tight_layout()
    return plt.gcf()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_logreg_experiments(path="heart_disease.csv", save_outputs=True, show_plots=True):
    data = run_pipeline(path)
    X, y = data["X"], data["y"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = get_70_30_split(X, y)

    section("1. Basic Logistic Regression (70/30 split)")
    basic_results = tune_and_evaluate(
        build_basic_pipeline(), BASIC_PARAM_GRID,
        X_train, X_test, y_train, y_test, "1. Basic Logistic Regression",
    )
    print(f"Best Params: {basic_results['metrics']['Best Params']}")
    print(f"Best CV F1: {basic_results['metrics']['CV F1 (best)']}")
    print(basic_results["classification_report"])

    section("2. SMOTE Logistic Regression (70/30 split)")
    smote_results = tune_and_evaluate(
        build_smote_pipeline(), SMOTE_PARAM_GRID,
        X_train, X_test, y_train, y_test, "2. SMOTE Logistic Regression",
    )
    print(f"Best Params: {smote_results['metrics']['Best Params']}")
    print(f"Best CV F1: {smote_results['metrics']['CV F1 (best)']}")
    print(smote_results["classification_report"])

    all_results = {
        "1. Basic Logistic Regression": basic_results,
        "2. SMOTE Logistic Regression": smote_results,
    }
    results_df = pd.DataFrame([basic_results["metrics"], smote_results["metrics"]])

    section("Comparison: Basic vs SMOTE Logistic Regression (70/30 split)")
    print(results_df.drop(columns=["Best Params"]).to_string(index=False))

    basic_coef_df = get_coefficients(basic_results["best_model"], feature_names)
    smote_coef_df = get_coefficients(smote_results["best_model"], feature_names)

    fig_cm = plot_confusion_matrices(all_results)
    fig_roc = plot_roc_curves(all_results, y_test)
    fig_metrics = plot_metric_comparison(results_df)
    fig_coef_basic = plot_coefficients(basic_coef_df, "Basic Logistic Regression -- Feature Influence")
    fig_coef_smote = plot_coefficients(smote_coef_df, "SMOTE Logistic Regression -- Feature Influence")
    if show_plots:
        plt.show()

    if save_outputs:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        results_df.to_csv(os.path.join(OUTPUT_DIR, "logreg_metrics_comparison.csv"), index=False)
        basic_coef_df.to_csv(os.path.join(OUTPUT_DIR, "logreg_basic_coefficients.csv"), index=False)
        smote_coef_df.to_csv(os.path.join(OUTPUT_DIR, "logreg_smote_coefficients.csv"), index=False)

        json_ready = {
            name: {
                "metrics": res["metrics"],
                "classification_report": res["classification_report"],
                "confusion_matrix": res["confusion_matrix"].tolist(),
            }
            for name, res in all_results.items()
        }
        with open(os.path.join(OUTPUT_DIR, "logreg_metrics.json"), "w") as f:
            json.dump(json_ready, f, indent=2)

        fig_cm.savefig(os.path.join(OUTPUT_DIR, "logreg_confusion_matrices.png"), dpi=150, bbox_inches="tight")
        fig_roc.savefig(os.path.join(OUTPUT_DIR, "logreg_roc_curves.png"), dpi=150, bbox_inches="tight")
        fig_metrics.savefig(os.path.join(OUTPUT_DIR, "logreg_metric_comparison.png"), dpi=150, bbox_inches="tight")
        fig_coef_basic.savefig(os.path.join(OUTPUT_DIR, "logreg_basic_coefficients.png"), dpi=150, bbox_inches="tight")
        fig_coef_smote.savefig(os.path.join(OUTPUT_DIR, "logreg_smote_coefficients.png"), dpi=150, bbox_inches="tight")
        print(f"\nSaved metrics + plots to ./{OUTPUT_DIR}/")

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "results": all_results,
        "results_df": results_df,
        "coefficients": {
            "1. Basic Logistic Regression": basic_coef_df,
            "2. SMOTE Logistic Regression": smote_coef_df,
        },
    }


if __name__ == "__main__":
    run_logreg_experiments()
