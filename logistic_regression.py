#Import Library
import os
import json
import math

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

#Constant
TEST_SIZE = 0.30 #30% for testing
OUTPUT_DIR = "outputs"

#C-values
_C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100] #C parameter
#Class weight
_CLASS_WEIGHTS = [None, "balanced"]

#Creating the Hyperparameter Grid
def _logreg_param_grid(prefix="logreg"): #creates the hyperparameter combinations
    return [
        {
            f"{prefix}__solver": ["liblinear"],
            f"{prefix}__penalty": ["l1", "l2"],
            f"{prefix}__C": _C_VALUES,
            f"{prefix}__class_weight": _CLASS_WEIGHTS,
        },
        #This tells GridSearchCV: Try :solver:liblinear, penalty:L1,L2, C:0.001,0.01,0.1,1,10,100, class weight:None,balanced
        #So it tries different combinations
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
        #solver = lbfgs, penalty = L2
        #Only L2 is used because lbfgs does not support L1 in this setup
    ]

#Creating the Two Parameter Grids
BASIC_PARAM_GRID = _logreg_param_grid("logreg") #Standard Scaler -> Logistic Regression
SMOTE_PARAM_GRID = _logreg_param_grid("logreg") #Standard Scaler -> SMOTE -> Logistic Regression
#Graph Settings
sns.set_style("whitegrid") #Sets the Seaborn graph style
plt.rcParams["figure.figsize"] = (7, 5) #Sets the default figure size

#Creates a formatting function for your console output
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
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y #random_state = Makes the split reproducible, stratify=y = ensures the training and testing sets maintain approximately the same class proportions
    )


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
def build_basic_pipeline(): #Original Data -> Standard Scaler -> Logistic Regression -> Prediction
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(random_state=RANDOM_STATE, max_iter=5000)),
    ])

#SMOTE Pipeline
def build_smote_pipeline(): ##Original Data -> Standard Scaler -> SMOTE -> Logistic Regression -> Prediction
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("logreg", LogisticRegression(random_state=RANDOM_STATE, max_iter=5000)),
    ])


# --------------------------------------------------------------------------
# Tune + evaluate
# --------------------------------------------------------------------------
def tune_and_evaluate(pipeline, param_grid, X_train, X_test, y_train, y_test, model_name, cv=5):
    n_jobs = min(4, os.cpu_count() or 1) #how many CPU workers GridSearchCV can use, It limits the number to a maximum of 4
    grid = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="f1", n_jobs=n_jobs, verbose=1,
    )
    grid.fit(X_train, y_train) #trains all the candidate models using the training data
    best_model = grid.best_estimator_ #Gets the best model found by GridSearchCV
    #Prediction
    y_pred = best_model.predict(X_test)
    #Probability Prediction
    y_prob = best_model.predict_proba(X_test)[:, 1] #gets the probability of the positive class
    #Creating Evaluation Metrics
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
    coefs = best_model.named_steps[step_name].coef_[0] #Gets the Logistic Regression coefficients, a coefficient tells you how a feature affects the predicted outcome
    names = list(feature_names)
    if len(names) != len(coefs):
        raise ValueError(
            f"Logistic Regression has {len(coefs)} coefficients but "
            f"{len(names)} feature names were passed. The cached model was "
            "likely fit before the one-hot encoding change. Clear "
            ".model_cache and rerun the app."
        )

    coef_df = pd.DataFrame({ #Creating the Coefficient DataFrame
        "Feature": names,
        "Coefficient": coefs,
    })
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs() #Creating the Absolute Coefficient, which features have the strongest influence regardless of direction
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True) #Ranking Features
    coef_df.insert(0, "Rank", range(1, len(coef_df) + 1))
    coef_df["Coefficient"] = coef_df["Coefficient"].round(4)
    coef_df["Abs_Coefficient"] = coef_df["Abs_Coefficient"].round(4)
    coef_df["Effect"] = np.where( #Determining the Effect
        coef_df["Coefficient"] > 0, "Increases risk", "Decreases risk"
    )
    return coef_df

#Plotting Coefficients
def plot_coefficients(coef_df, title="Logistic Regression Coefficients (Feature Influence)"): #Creates a horizontal bar graph showing feature influence
    plot_df = coef_df.sort_values("Coefficient") #Sorts features according to their coefficient values
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in plot_df["Coefficient"]]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["Feature"], plot_df["Coefficient"], color=colors) #creates the horizontal bars
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Coefficient (negative = lowers risk, positive = raises risk)")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Visualizations
# --------------------------------------------------------------------------
#Confusion Matrix Plot
def plot_confusion_matrices(all_results, class_labels=("No", "Yes")):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, all_results.items()):
        sns.heatmap(
            res["confusion_matrix"], annot=True, fmt="d", cmap="Blues", #annot=True means the actual numbers are displayed
            xticklabels=class_labels, yticklabels=class_labels, cbar=False, ax=ax,
        )
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle(f"Confusion Matrices ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split)", fontsize=14)
    plt.tight_layout()
    return fig

#ROC Curves for Basic and SMOTE
def plot_roc_curves(all_results, y_test):
    plt.figure(figsize=(7, 6))
    for name, res in all_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"]) #calculate falso positive rate and true positive rate
        auc = res["metrics"]["ROC-AUC"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})") #Plots each model's ROC curve
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance") #Creates the diagonal reference line = represents approximately random classification
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: Basic vs SMOTE Logistic Regression ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt.gcf()

#Metric Comparison
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
        plt.gca().bar_label(container, fmt=lambda v: f"{math.floor(v * 100 + 0.5) / 100:.2f}", fontsize=8, padding=2)
    plt.tight_layout()
    return plt.gcf()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
#Main Experiment Function
def run_logreg_experiments(path="heart_disease.csv", save_outputs=True, show_plots=True):
    data = run_pipeline(path) #Calls your preprocessing pipeline
    X, y = data["X"], data["y"] #x = feature, y = target
    feature_names = X.columns.tolist() #Gets the names of all input features

    X_train, X_test, y_train, y_test = get_70_30_split(X, y)

    section("1. Basic Logistic Regression (70/30 split)") #Create 70/30 Dataset
    basic_results = tune_and_evaluate( #Train Basic Logistic Regression
        build_basic_pipeline(), BASIC_PARAM_GRID,
        X_train, X_test, y_train, y_test, "1. Basic Logistic Regression",
    )
    print(f"Best Params: {basic_results['metrics']['Best Params']}")
    print(f"Best CV F1: {basic_results['metrics']['CV F1 (best)']}")
    print(basic_results["classification_report"])

    section("2. SMOTE Logistic Regression (70/30 split)") #Train SMOTE Logistic Regression
    smote_results = tune_and_evaluate(
        build_smote_pipeline(), SMOTE_PARAM_GRID,
        X_train, X_test, y_train, y_test, "2. SMOTE Logistic Regression",
    )
    print(f"Best Params: {smote_results['metrics']['Best Params']}")
    print(f"Best CV F1: {smote_results['metrics']['CV F1 (best)']}")
    print(smote_results["classification_report"])

    all_results = { #Store Both Models
        "1. Basic Logistic Regression": basic_results,
        "2. SMOTE Logistic Regression": smote_results,
    }
    results_df = pd.DataFrame([basic_results["metrics"], smote_results["metrics"]]) #Create Comparison Table

    section("Comparison: Basic vs SMOTE Logistic Regression (70/30 split)")
    print(results_df.drop(columns=["Best Params"]).to_string(index=False))

    basic_coef_df = get_coefficients(basic_results["best_model"], feature_names)
    smote_coef_df = get_coefficients(smote_results["best_model"], feature_names)

    fig_cm = plot_confusion_matrices(all_results) #Creates confusion matrix graphs
    fig_roc = plot_roc_curves(all_results, y_test) #Creates ROC curves
    fig_metrics = plot_metric_comparison(results_df) #Creates metric comparison
    fig_coef_basic = plot_coefficients(basic_coef_df, "Basic Logistic Regression -- Feature Influence") #creates the Basic Logistic Regression coefficient graph
    fig_coef_smote = plot_coefficients(smote_coef_df, "SMOTE Logistic Regression -- Feature Influence") #creates the SMOTE coefficient graph
    if show_plots: #Display graph
        plt.show()

    #Create Output Folder
    if save_outputs:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        results_df.to_csv(os.path.join(OUTPUT_DIR, "logreg_metrics_comparison.csv"), index=False) #Save Metrics
        basic_coef_df.to_csv(os.path.join(OUTPUT_DIR, "logreg_basic_coefficients.csv"), index=False) #Save Coefficients
        smote_coef_df.to_csv(os.path.join(OUTPUT_DIR, "logreg_smote_coefficients.csv"), index=False) #Save Coefficients

        json_ready = { #Save JSON Results
            name: {
                "metrics": res["metrics"],
                "classification_report": res["classification_report"],
                "confusion_matrix": res["confusion_matrix"].tolist(),
            }
            for name, res in all_results.items()
        }
        with open(os.path.join(OUTPUT_DIR, "logreg_metrics.json"), "w") as f:
            json.dump(json_ready, f, indent=2)

        #Save Graphs
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
