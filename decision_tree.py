"""
Decision Tree classifier for Heart Disease Status (Yes / No).

A single tree is used as a non-linear, threshold-based model that does not
assume a linear relationship between predictors and the outcome (report
Section 5.3). It also provides the single-tree baseline that Random Forest
is compared against.

Two pipelines are tuned the same way, plus a Dummy (always-No) baseline:
    0. Dummy  -- majority class; 80% accuracy, recall 0, AUC 0.50
    1. Basic  -- original 4:1 class distribution
    2. SMOTE  -- synthetic oversampling of the minority "Yes" class on train folds only

Hyperparameters are chosen by stratified 5-fold GridSearchCV with F1-score,
then each best model is scored once on the held-out 30% test set.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

from data_preprocessing import run_pipeline, RANDOM_STATE

TEST_SIZE = 0.30  # 7,000 train / 3,000 test
OUTPUT_DIR = "outputs"

# F1 (not accuracy): a tree that predicts all "No" would score 80% accuracy
# on this 4:1 dataset while missing every heart-disease case.
DT_SCORING = "f1"

# Shallow trees only. Depth 3/5/8 and min_samples_leaf 10/25/50 limit
# overfitting on 7,000 training rows. gini vs entropy is the split criterion.
BASIC_PARAM_GRID = {
    "dt__criterion": ["gini", "entropy"],
    "dt__max_depth": [3, 5, 8],
    "dt__min_samples_leaf": [10, 25, 50],
}
# Same grid as Basic so Basic vs SMOTE isolates the effect of oversampling,
# not a change in search space.
SMOTE_PARAM_GRID = {
    "dt__criterion": ["gini", "entropy"],
    "dt__max_depth": [3, 5, 8],
    "dt__min_samples_leaf": [10, 25, 50],
}

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7, 5)


def section(title):
    """Print a centred banner so terminal output is easy to scan."""
    width = 70
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

def get_70_30_split(X, y, random_state=RANDOM_STATE):
    """Stratified 70/30 train-test split, matching knn.py / logreg.py / rf.py.

    Stratify keeps the 80/20 class ratio in both sets. random_state=42 makes
    the same 7,000 / 3,000 rows used by every model.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------

def build_basic_pipeline():
    # Trees split on feature thresholds, so they do not need scaling.
    return ImbPipeline([
        ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])


def build_smote_pipeline():
    # SMOTE interpolates using Euclidean distance, so features are scaled first
    # even though the tree itself is scale-invariant. SMOTE runs only on the
    # training fold inside GridSearchCV (ImbPipeline), never on the test set.
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])


# --------------------------------------------------------------------------
# Tune + evaluate
# --------------------------------------------------------------------------

def tune_and_evaluate(pipeline, param_grid, X_train, X_test, y_train, y_test, model_name, cv=5, scoring=DT_SCORING):
    """Fit GridSearchCV on train, then score the winner once on the test set.

    Test metrics are not used to pick hyperparameters. ROC-AUC uses predict
    probabilities so it is threshold-independent (unlike F1 / recall at 0.5).
    """
    # n_jobs=-1 uses all CPU cores; cv=5 is stratified via the default splitter
    # because y is binary and class counts are preserved in each fold.
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

    y_pred = best_model.predict(X_test)                 # class labels at threshold 0.5
    y_prob = best_model.predict_proba(X_test)[:, 1]     # P(Heart Disease = Yes)

    # zero_division=0: if the tree predicts no "Yes" cases, precision is 0 not NaN.
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


def evaluate_dummy(X_train, X_test, y_train, y_test, model_name="Dummy (always No)"):
    """Majority-class baseline: always predict No (80% of this dataset).

    Accuracy looks strong; recall / F1 for Yes are 0; ROC-AUC is 0.50 because
    every row gets the same predicted probability. This is the bar a real
    model has to beat -- not 80% accuracy on its own.
    """
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)

    y_pred = dummy.predict(X_test)
    y_prob = dummy.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Split": f"{int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}",
        "Best Params": {"strategy": "most_frequent"},
        "CV Scoring Metric": "n/a",
        f"CV {DT_SCORING.upper()} (best)": "n/a",
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }

    return {
        "grid": None,
        "best_model": dummy,
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
    """Permutation importance on the test set (report Figure 5.8).

    Shuffle one feature at a time and measure the drop in F1 (same metric the
    trees are tuned on). Gini impurity importance is not used: it can favour
    high-cardinality features. Negative values mean shuffling that feature
    did not hurt (or slightly helped) F1.
    """
    result = permutation_importance(
        model, X_test, y_test, scoring="f1",
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": result.importances_mean,
        "Std_Dev": result.importances_std,
    })

    imp_df = imp_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    imp_df.index += 1
    imp_df.index.name = "Rank"

    return imp_df.reset_index()


def plot_permutation_importance(imp_df, title, top_n=None):
    """Horizontal bar chart of permutation importance (all features, or top_n)."""
    plot_df = imp_df if top_n is None else imp_df.head(top_n)

    # Grow the figure with the number of bars so the labels stay readable.
    height = max(6, 0.32 * len(plot_df))
    fig, ax = plt.subplots(figsize=(8, height))

    sns.barplot(
        x="Importance",
        y="Feature",
        hue="Feature",
        data=plot_df,
        palette="viridis",
        legend=False,
        ax=ax,
    )

    ax.axvline(0, color="grey", linewidth=1)  # zero = shuffling did not change F1
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Mean F1 Decrease (Importance)")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig


def plot_confusion_matrices(all_results, class_labels=("No", "Yes")):
    """Side-by-side heatmaps: Dummy vs Basic vs SMOTE (true Neg/Pos vs predicted)."""
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
    """ROC on the same test set. The dashed diagonal is chance (AUC = 0.50)."""
    plt.figure(figsize=(7, 6))
    for name, res in all_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        auc = res["metrics"]["ROC-AUC"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt.gcf()


def plot_metric_comparison(results_df):
    """Grouped bars for Accuracy, Precision, Recall, F1, and ROC-AUC."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    plot_df = results_df.melt(
        id_vars="Model", value_vars=metric_cols,
        var_name="Metric", value_name="Score",
    )
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="Set2")
    plt.ylim(0, 1)
    plt.title("Decision Tree Metric Comparison (70/30 split)")
    plt.legend(title=None, loc="lower right")
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.tight_layout()
    return plt.gcf()


def plot_fitted_tree(model, feature_names, class_names=("No", "Yes"), max_plot_depth=4):
    """Draw the top levels of the fitted Basic tree (report interpretability slide).

    The tuned Basic tree is depth 8, which is too wide to read as a figure.
    Only the first max_plot_depth levels are drawn; export_text() writes the
    full set of rules. Weak / noisy splits here are consistent with AUC ~ 0.50.
    """
    tree = model.named_steps["dt"] if hasattr(model, "named_steps") else model
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        tree,
        feature_names=list(feature_names),
        class_names=list(class_names),
        filled=True,
        rounded=True,
        fontsize=7,
        max_depth=max_plot_depth,
        ax=ax,
    )
    ax.set_title(
        f"Basic Decision Tree — first {max_plot_depth} of {tree.get_depth()} levels",
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_decision_tree_experiments(path="heart_disease.csv", save_outputs=True, show_plots=True):
    """Run Dummy vs Basic vs SMOTE Decision Tree and write tables/plots to outputs/."""
    # Clean + encode in memory; heart_disease.csv on disk stays raw.
    data = run_pipeline(path)
    X, y = data["X"], data["y"]

    X_train, X_test, y_train, y_test = get_70_30_split(X, y)

    # Pipeline 1: original class balance (8,000 No / 2,000 Yes overall).
    section("1. Basic Decision Tree (70/30 split)")
    basic_results = tune_and_evaluate(
        build_basic_pipeline(), BASIC_PARAM_GRID,
        X_train, X_test, y_train, y_test, "1. Basic Decision Tree",
        scoring=DT_SCORING,
    )
    print(f"Best Params: {basic_results['metrics']['Best Params']}")
    print(f"Best CV {DT_SCORING.upper()}: {basic_results['metrics'][f'CV {DT_SCORING.upper()} (best)']}")
    print(basic_results["classification_report"])

    # Pipeline 2: SMOTE on training folds only; test set is never resampled.
    section("2. SMOTE Decision Tree (70/30 split)")
    smote_results = tune_and_evaluate(
        build_smote_pipeline(), SMOTE_PARAM_GRID,
        X_train, X_test, y_train, y_test, "2. SMOTE Decision Tree",
        scoring=DT_SCORING,
    )
    print(f"Best Params: {smote_results['metrics']['Best Params']}")
    print(f"Best CV {DT_SCORING.upper()}: {smote_results['metrics'][f'CV {DT_SCORING.upper()} (best)']}")
    print(smote_results["classification_report"])

    # Majority-class baseline: always predict No. Not tuned.
    section("3. Dummy (always No)")
    dummy_results = evaluate_dummy(X_train, X_test, y_train, y_test)
    print(dummy_results["classification_report"])

    all_results = {
        "Dummy (always No)": dummy_results,
        "1. Basic Decision Tree": basic_results,
        "2. SMOTE Decision Tree": smote_results,
    }
    results_df = pd.DataFrame([
        dummy_results["metrics"],
        basic_results["metrics"],
        smote_results["metrics"],
    ])

    section("Comparison: Dummy vs Basic DT vs SMOTE DT (70/30 split)")
    print(results_df.drop(columns=["Best Params"]).to_string(index=False))

    section("Permutation importance (test set, F1 drop)")
    imp_basic = get_permutation_importance(basic_results["best_model"], X_test, y_test)
    imp_smote = get_permutation_importance(smote_results["best_model"], X_test, y_test)
    print("Basic DT (top 10):")
    print(imp_basic.head(10).to_string(index=False))
    print("\nSMOTE DT (top 10):")
    print(imp_smote.head(10).to_string(index=False))

    fig_cm = plot_confusion_matrices(all_results)
    fig_roc = plot_roc_curves(all_results, y_test)
    fig_metrics = plot_metric_comparison(results_df)
    fig_imp_basic = plot_permutation_importance(imp_basic, "Basic Decision Tree -- Permutation Importance (F1)")
    fig_imp_smote = plot_permutation_importance(imp_smote, "SMOTE Decision Tree -- Permutation Importance (F1)")
    fig_tree = plot_fitted_tree(basic_results["best_model"], X_train.columns)
    if show_plots:
        plt.show()

    if save_outputs:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        results_df.to_csv(os.path.join(OUTPUT_DIR, "decision_tree_metrics_comparison.csv"), index=False)
        imp_basic.to_csv(os.path.join(OUTPUT_DIR, "decision_tree_permutation_importance_basic.csv"), index=False)
        imp_smote.to_csv(os.path.join(OUTPUT_DIR, "decision_tree_permutation_importance_smote.csv"), index=False)

        tree_text = export_text(
            basic_results["best_model"].named_steps["dt"],
            feature_names=list(X_train.columns),
            decimals=3,
        )
        with open(os.path.join(OUTPUT_DIR, "decision_tree_basic_rules.txt"), "w", encoding="utf-8") as f:
            f.write(tree_text)

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
        fig_imp_basic.savefig(os.path.join(OUTPUT_DIR, "decision_tree_permutation_importance_basic.png"), dpi=150, bbox_inches="tight")
        fig_imp_smote.savefig(os.path.join(OUTPUT_DIR, "decision_tree_permutation_importance_smote.png"), dpi=150, bbox_inches="tight")
        fig_tree.savefig(os.path.join(OUTPUT_DIR, "decision_tree_basic_structure.png"), dpi=150, bbox_inches="tight")
        print(f"\nSaved metrics + plots to ./{OUTPUT_DIR}/")


    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "results": all_results,
        "results_df": results_df,
        "permutation_importance": {"basic": imp_basic, "smote": imp_smote},
    }


if __name__ == "__main__":
    run_decision_tree_experiments()
