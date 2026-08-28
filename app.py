import hashlib
import os

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import data_preprocessing as dp
import data_visualization as dv
import plotly.graph_objects as go
import knn as km
import decision_tree as dtm
import logistic_regression as lgm
import random_forest as rfm
import feature_selection_check as fscm
import pca_check as pcam

st.set_page_config(page_title="Heart Disease Risk", layout="wide", page_icon="\u2764\ufe0f")

st.html(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
    :root {
        --accent: #0f766e;
        --accent-dark: #0b5b54;
        --accent-soft: #e6f4f2;
        --pulse: #e11d48;
        --ink: #16232b;
        --muted: #64748b;
        --line: #e2e8f0;
        --card-radius: 14px;
    }

    /* ---- Typography: one clean face everywhere ---- */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    input, textarea, select, button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* ---- Canvas: soft neutral instead of stark white ---- */
    [data-testid="stAppViewContainer"] { background: #f8fafa; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .block-container { padding-top: 1.6rem; max-width: 1200px; }

    /* ---- Masthead: thin signature gradient bar, ties the two accent
       colors together at the very top of the page ---- */
    [data-testid="stDecoration"] {
        background: linear-gradient(90deg, var(--accent) 0%, var(--pulse) 100%) !important;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: #f1f5f4;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }
    /* Bug fix: Streamlit wraps each radio option's label text in its own
       <p>, and an explicit color on that <p> always wins over an inherited
       color from the ancestor <label> — so the label-level color rules
       below never actually reached the visible text. That made every
       unchecked nav item render in low-contrast muted gray on a near-white
       row (illegible). Target the <p> itself, at higher specificity than
       the blanket muted-text rule above. */
    [data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
        color: var(--ink) !important;
        font-weight: 500;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover [data-testid="stMarkdownContainer"] p {
        color: var(--accent-dark) !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 700;
    }
    /* Sidebar nav: hide the bare radio circle, style each option as a
       full-width row/pill so it reads as a menu, not a form field. */
    [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        display: flex;
        align-items: center;
        width: 100%;
        padding: 10px 14px;
        margin-bottom: 6px;
        border-radius: 10px;
        background-color: rgba(0, 0, 0, 0.025);
        transition: background 0.15s ease, color 0.15s ease;
        cursor: pointer;
        font-weight: 500;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: var(--accent-soft) !important;
        color: var(--accent-dark) !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: var(--accent) !important;
        color: #ffffff !important;
        font-weight: 700;
    }

    /* ---- Headings: one consistent accent, with real breathing room
       between sections instead of Streamlit's default cramped rhythm ---- */
    h1, h2, h3 {
        color: var(--ink) !important;
        letter-spacing: -0.01em;
        font-weight: 800 !important;
    }
    h1 { font-size: 2.3rem !important; border-bottom: 3px solid var(--accent); padding-bottom: 0.35rem; display: inline-block; }
    h2 { font-size: 1.4rem !important; margin-top: 2.2rem !important; }
    h3 { font-size: 1.1rem !important; margin-top: 1.4rem !important; }

    /* ---- Metric widgets -> small cards instead of bare numbers ---- */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: var(--card-radius);
        padding: 0.9rem 1rem 0.7rem 1rem;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.72rem !important;
    }

    /* ---- Tabs: navy + rose, not teal ---- */
    [data-testid="stTab"] { font-weight: 600; }
    [data-testid="stTab"][data-selected] { color: #1e3a4c !important; }
    [data-testid="stTab"][data-selected] .react-aria-SelectionIndicator {
        background-color: #e11d48 !important;
    }

    /* Equal-width split tabs (model / EDA / robustness category bars) */
    .st-key-eda_category_tabs [role="tablist"],
    .st-key-smote_model_tabs [role="tablist"],
    .st-key-rob_category_tabs [role="tablist"] {
        display: flex !important;
        width: 100%;
        gap: 0;
        background: #ffffff;
        border: 1px solid #d8dee6;
        border-radius: 12px;
        padding: 0;
        overflow: hidden;
    }
    .st-key-eda_category_tabs [data-testid="stTab"],
    .st-key-smote_model_tabs [data-testid="stTab"],
    .st-key-rob_category_tabs [data-testid="stTab"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        justify-content: center !important;
        height: 3.2rem !important;
        padding: 0 0.55rem !important;
        border-radius: 0 !important;
        background: #f4f6f8 !important;
        color: #5c6b7a !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-right: 1px solid #d8dee6 !important;
    }
    .st-key-eda_category_tabs [data-testid="stTab"]:last-of-type,
    .st-key-smote_model_tabs [data-testid="stTab"]:last-of-type,
    .st-key-rob_category_tabs [data-testid="stTab"]:last-of-type {
        border-right: none !important;
    }
    .st-key-eda_category_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"],
    .st-key-smote_model_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"],
    .st-key-rob_category_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"],
    .st-key-eda_category_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"] p,
    .st-key-smote_model_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"] p,
    .st-key-rob_category_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"] p,
    .st-key-eda_category_tabs [data-testid="stTab"] [data-testid="stIconMaterial"],
    .st-key-smote_model_tabs [data-testid="stTab"] [data-testid="stIconMaterial"],
    .st-key-rob_category_tabs [data-testid="stTab"] [data-testid="stIconMaterial"] {
        font-size: inherit !important;
        font-weight: 700 !important;
        color: inherit !important;
        text-align: center;
        width: 100%;
    }
    .st-key-eda_category_tabs [data-testid="stTab"][data-hovered]:not([data-selected]),
    .st-key-smote_model_tabs [data-testid="stTab"][data-hovered]:not([data-selected]),
    .st-key-rob_category_tabs [data-testid="stTab"][data-hovered]:not([data-selected]) {
        background: #e8edf2 !important;
        color: #1e3a4c !important;
    }
    .st-key-eda_category_tabs [data-testid="stTab"][data-selected],
    .st-key-smote_model_tabs [data-testid="stTab"][data-selected],
    .st-key-rob_category_tabs [data-testid="stTab"][data-selected],
    .st-key-eda_category_tabs [data-testid="stTab"][aria-selected="true"],
    .st-key-smote_model_tabs [data-testid="stTab"][aria-selected="true"],
    .st-key-rob_category_tabs [data-testid="stTab"][aria-selected="true"] {
        background: #1e3a4c !important;
        color: #ffffff !important;
    }
    .st-key-eda_category_tabs [data-testid="stTab"] .react-aria-SelectionIndicator,
    .st-key-smote_model_tabs [data-testid="stTab"] .react-aria-SelectionIndicator,
    .st-key-rob_category_tabs [data-testid="stTab"] .react-aria-SelectionIndicator {
        display: none !important;
    }

    /* ---- Expanders and bordered containers: shared card radius ---- */
    [data-testid="stExpander"] {
        border-radius: var(--card-radius);
        border: 1px solid var(--line);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: var(--card-radius); }

    /* ---- Alerts: a left accent stripe instead of a flat default box,
       colored per severity so warning/success/error stay legible ---- */
    [data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid var(--line);
        border-left-width: 5px;
        border-left-style: solid;
    }
    [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) { border-left-color: #d97706; }
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) { border-left-color: var(--accent); }
    [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) { border-left-color: #16a34a; }
    [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) { border-left-color: var(--pulse); }

    /* ---- Progress bar in the accent color ---- */
    [data-testid="stProgress"] > div > div > div { background-color: var(--accent) !important; }

    /* ---- Buttons in the accent color, not Streamlit's default red ---- */
    button[kind="primary"], button[kind="formSubmit"] {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
        border-radius: 10px !important;
    }
    button[kind="primary"]:hover, button[kind="formSubmit"]:hover {
        background-color: var(--accent-dark) !important;
        border-color: var(--accent-dark) !important;
    }

    /* ---- Section breathing room ---- */
    [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] { margin-bottom: 0.15rem; }
    </style>
    """
)

PULSE_DIVIDER_SVG = """
<svg width="220" height="20" viewBox="0 0 220 20" xmlns="http://www.w3.org/2000/svg" style="margin: 2px 0 10px 0;">
  <polyline points="0,10 55,10 65,2 75,18 85,10 95,10 102,4 109,16 116,10 220,10"
    fill="none" stroke="#0f766e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

DEFAULT_DATA_PATH = "heart_disease.csv"
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
# Bump this when encoding / feature columns change so Streamlit and joblib
# caches do not reuse models fit on the previous column set.
FEATURE_CACHE_TAG = "keep_both_dummies"


def _feature_cache_sig(X):
    joined = "|".join(map(str, X.columns))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]


def load_or_train(cache_name, compute_fn, X, y):
    sig = _feature_cache_sig(X)
    path = os.path.join(MODEL_CACHE_DIR, f"{cache_name}_{sig}.joblib")
    if os.path.exists(path):
        data = joblib.load(path)
        model = data.get("result", {}).get("best_model")
        n_in = getattr(model, "n_features_in_", None) if model is not None else None
        if n_in is None or n_in == X.shape[1]:
            return data
        os.remove(path)
    data = compute_fn(X, y)
    joblib.dump(data, path)
    return data


def load_or_compute(cache_name, compute_fn):
    """Disk-cache any picklable result (DataFrame, dict, ...) under a fixed
    name so it survives Streamlit restarts, not just the current session.
    Used for the heavier one-off checks (robustness sweep, ANOVA feature
    selection, PCA) that don't need the feature-signature invalidation that
    load_or_train uses for the four main models."""
    path = os.path.join(MODEL_CACHE_DIR, f"{cache_name}_{FEATURE_CACHE_TAG}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    result = compute_fn()
    joblib.dump(result, path)
    return result


def with_spinner(label, func, *args, **kwargs):
    """Show a spinner with a specific label while a (possibly cached)
    function runs, so the person always sees what's currently happening
    instead of a frozen page."""
    with st.spinner(label):
        return func(*args, **kwargs)


def pick_recommended_model(df):
    """Mirrors the report's Section 6.1 reasoning, not a bare ROC-AUC argmax:
    Basic KNN has the nominally highest test-set ROC-AUC, but the margin over
    Random Forest is within noise. The report selects Random Forest as the
    more defensible choice instead — more robust to the class imbalance
    (balanced class weighting), more interpretable (permutation importance),
    and more consistent across the Section 5.6 robustness checks. Returns
    (recommended_row, nominal_auc_leader_row); falls back to the ROC-AUC
    leader if Random Forest isn't present in the given table."""
    auc_leader_row = df.loc[df["ROC-AUC"].idxmax()]
    rf_rows = df[df["Model"] == "Random Forest"]
    recommended_row = rf_rows.iloc[0] if not rf_rows.empty else auc_leader_row
    return recommended_row, auc_leader_row


_METRIC_KPI_SPEC = [
    ("acc", "green", ":material/verified:", "Accuracy"),
    ("prec", "red", ":material/filter_alt:", "Precision"),
    ("rec", "blue", ":material/track_changes:", "Recall"),
    ("f1", "orange", ":material/balance:", "F1-Score"),
    ("auc", "green", ":material/monitoring:", "ROC-AUC"),
]


def render_smote_delta_kpis(prefix, row, delta_fn):
    """Five colored metric cards. `row` is the displayed pipeline; the pill is SMOTE − Basic."""
    for col, (suffix, color, icon, label) in zip(st.columns(5), _METRIC_KPI_SPEC):
        with col:
            with st.container(border=True, key=f"{prefix}_{suffix}"):
                st.metric(
                    f"{icon} {label}",
                    f":{color}[**{row[label]:.4f}**]",
                    f"{delta_fn(label):.4f}",
                )


@st.cache_data(show_spinner="Loading & preprocessing data...")
def load_pipeline_data(path):
    return dp.run_pipeline(path)


@st.cache_data(show_spinner="Loading raw data...")
def load_raw_data(path):
    df = dp.load_data(path)
    numeric_cols, categorical_cols = dp.get_column_groups(df)
    return df, numeric_cols, categorical_cols


@st.cache_resource(show_spinner=False)
def train_knn_basic(X, y, _cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = km.get_70_30_split(X, y)
        result = km.tune_and_evaluate(
            km.build_basic_pipeline(), km.BASIC_PARAM_GRID,
            X_train, X_test, y_train, y_test, "1. Basic KNN",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("knn_basic", _compute, X, y)


@st.cache_resource(show_spinner=False)
def train_knn_smote(X, y, _cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = km.get_70_30_split(X, y)
        result = km.tune_and_evaluate(
            km.build_smote_pipeline(), km.SMOTE_PARAM_GRID,
            X_train, X_test, y_train, y_test, "2. SMOTE KNN",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("knn_smote", _compute, X, y)


@st.cache_resource(show_spinner=False)
def train_dt_basic(X, y, _cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = dtm.get_70_30_split(X, y)
        result = dtm.tune_and_evaluate(
            dtm.build_basic_pipeline(), dtm.BASIC_PARAM_GRID,
            X_train, X_test, y_train, y_test, "1. Basic Decision Tree",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("dt_basic", _compute, X, y)


@st.cache_resource(show_spinner=False)
def train_dt_smote(X, y, _cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = dtm.get_70_30_split(X, y)
        result = dtm.tune_and_evaluate(
            dtm.build_smote_pipeline(), dtm.SMOTE_PARAM_GRID,
            X_train, X_test, y_train, y_test, "2. SMOTE Decision Tree",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("dt_smote", _compute, X, y)


@st.cache_resource(show_spinner=False)
def train_lr_basic(X, y, _cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = lgm.get_70_30_split(X, y)
        result = lgm.tune_and_evaluate(
            lgm.build_basic_pipeline(), lgm.BASIC_PARAM_GRID,
            X_train, X_test, y_train, y_test, "1. Basic Logistic Regression",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("lr_basic", _compute, X, y)


@st.cache_resource(show_spinner=False)
def train_lr_smote(X, y, _cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = lgm.get_70_30_split(X, y)
        result = lgm.tune_and_evaluate(
            lgm.build_smote_pipeline(), lgm.SMOTE_PARAM_GRID,
            X_train, X_test, y_train, y_test, "2. SMOTE Logistic Regression",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("lr_smote", _compute, X, y)


@st.cache_resource(show_spinner=False)
def train_rf_basic(X, y, _cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = rfm.get_70_30_split(X, y)
        result = rfm.tune_and_evaluate(
            rfm.build_basic_pipeline(), rfm.BASIC_PARAM_GRID,
            X_train, X_test, y_train, y_test, "1. Basic Random Forest",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("rf_basic", _compute, X, y)


@st.cache_resource(show_spinner=False)
def train_rf_smote(X, y, _cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = rfm.get_70_30_split(X, y)
        result = rfm.tune_and_evaluate(
            rfm.build_smote_pipeline(), rfm.SMOTE_PARAM_GRID,
            X_train, X_test, y_train, y_test, "2. SMOTE Random Forest",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("rf_smote", _compute, X, y)

def run_training_jobs(label, jobs):
    """Train each job with a single tidy progress bar instead of a long
    scrolling checklist — shows what's running right now without spamming
    16 lines of history once everything's done."""
    outputs = {}
    total = len(jobs)
    progress = st.progress(0.0, text=f"{label}...")
    for i, (name, (func, X, y)) in enumerate(jobs.items(), start=1):
        progress.progress((i - 1) / total, text=f"{label} — training {name} ({i}/{total})...")
        outputs[name] = func(X, y)
    progress.progress(1.0, text=f"{label} — done ✅")
    return outputs


ROBUSTNESS_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=dp.RANDOM_STATE)


def _eval_pipeline(label, scoring_label, pipeline, param_grid, scoring, X_train, X_test, y_train, y_test):
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(pipeline, param_grid, cv=ROBUSTNESS_CV, scoring=scoring, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    row = {
        "Check": label,
        "Scoring": scoring_label,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }
    return row, best, y_prob


def _best_threshold_from_cv(pipeline, X_train, y_train):
    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=ROBUSTNESS_CV, method="predict_proba")[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    return thresholds[np.nanargmax(f1s[:-1])]


def _add_interaction_terms(X):
    X = X.copy()
    X["Age_x_BMI"] = X["Age"] * X["BMI"]
    X["Cholesterol_x_BloodPressure"] = X["Cholesterol Level"] * X["Blood Pressure"]
    X["Sleep_x_Stress"] = X["Sleep Hours"] * X["Stress Level"]
    X["Triglyceride_x_FastingBloodSugar"] = X["Triglyceride Level"] * X["Fasting Blood Sugar"]
    X["CRP_x_Homocysteine"] = X["CRP Level"] * X["Homocysteine Level"]
    return X


@st.cache_resource(show_spinner=False)
def run_robustness_checks(X, y):
    return load_or_compute("robustness_checks", lambda: _compute_robustness_checks(X, y))


def _compute_robustness_checks(X, y):
    from sklearn.model_selection import train_test_split
    X_eng = _add_interaction_terms(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=dp.RANDOM_STATE, stratify=y)
    X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(X_eng, y, test_size=0.30, random_state=dp.RANDOM_STATE, stratify=y)

    dt_grid_current = {"dt__criterion": ["gini", "entropy"], "dt__max_depth": [3, 5, 8], "dt__min_samples_leaf": [10, 25, 50]}
    dt_grid_widened = {"dt__criterion": ["gini", "entropy"], "dt__max_depth": [3, 5, 8, 12, None], "dt__min_samples_leaf": [5, 10, 25, 50], "dt__min_samples_split": [2, 10, 20], "dt__class_weight": [None, "balanced"]}
    rf_grid_current = {"rf__n_estimators": [100, 300], "rf__min_samples_leaf": [5, 10], "rf__class_weight": ["balanced"]}
    rf_grid_free_weight = {"rf__n_estimators": [100, 300], "rf__min_samples_leaf": [5, 10], "rf__class_weight": [None, "balanced"]}

    rows = []
    stages = ["scoring=f1", "scoring=accuracy", "scoring=roc_auc", "scoring=recall",
              "threshold tuning", "widened grids", "interaction features"]
    total_stages = len(stages)
    progress = st.progress(0.0, text="Running robustness checks...")

    for stage_i, scoring in enumerate(["f1", "accuracy", "roc_auc", "recall"]):
        progress.progress(stage_i / total_stages, text=f"Robustness checks — re-tuning all 4 models with {stages[stage_i]}...")
        row, _, _ = _eval_pipeline("KNN", f"scoring={scoring}", Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))]), {}, scoring, X_train, X_test, y_train, y_test)
        rows.append(row)
        row, _, _ = _eval_pipeline("Logistic Regression", f"scoring={scoring}", Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=5000, random_state=dp.RANDOM_STATE))]), {}, scoring, X_train, X_test, y_train, y_test)
        rows.append(row)
        row, _, _ = _eval_pipeline("Decision Tree", f"scoring={scoring}", Pipeline([("dt", DecisionTreeClassifier(random_state=dp.RANDOM_STATE))]), dt_grid_current, scoring, X_train, X_test, y_train, y_test)
        rows.append(row)
        row, _, _ = _eval_pipeline("Random Forest", f"scoring={scoring}", Pipeline([("rf", RandomForestClassifier(random_state=dp.RANDOM_STATE))]), rf_grid_current, scoring, X_train, X_test, y_train, y_test)
        rows.append(row)

    progress.progress(4 / total_stages, text="Robustness checks — tuning the decision threshold from cross-validated predictions...")
    row, dt_model, dt_prob = _eval_pipeline("Decision Tree", "f1 (baseline)", Pipeline([("dt", DecisionTreeClassifier(random_state=dp.RANDOM_STATE))]), dt_grid_current, "f1", X_train, X_test, y_train, y_test)
    tuned_t = _best_threshold_from_cv(dt_model, X_train, y_train)
    y_pred_t = (dt_prob >= tuned_t).astype(int)
    rows.append({
        "Check": "Decision Tree", "Scoring": f"f1 + threshold tuned (t={tuned_t:.3f})",
        "Accuracy": round(accuracy_score(y_test, y_pred_t), 4), "Precision": round(precision_score(y_test, y_pred_t, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred_t, zero_division=0), 4), "F1-Score": round(f1_score(y_test, y_pred_t, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, dt_prob), 4),
    })

    progress.progress(5 / total_stages, text="Robustness checks — re-tuning with widened hyperparameter grids...")
    row, _, _ = _eval_pipeline("Decision Tree", "widened grid (class_weight incl.)", Pipeline([("dt", DecisionTreeClassifier(random_state=dp.RANDOM_STATE))]), dt_grid_widened, "f1", X_train, X_test, y_train, y_test)
    rows.append(row)
    row, _, _ = _eval_pipeline("Random Forest", "class_weight=None allowed", Pipeline([("rf", RandomForestClassifier(random_state=dp.RANDOM_STATE))]), rf_grid_free_weight, "f1", X_train, X_test, y_train, y_test)
    rows.append(row)

    progress.progress(6 / total_stages, text="Robustness checks — re-tuning with engineered interaction features...")
    row, _, _ = _eval_pipeline("Decision Tree", "+ interaction features", Pipeline([("dt", DecisionTreeClassifier(random_state=dp.RANDOM_STATE))]), dt_grid_current, "f1", X_train_eng, X_test_eng, y_train_eng, y_test_eng)
    rows.append(row)
    row, _, _ = _eval_pipeline("Random Forest", "+ interaction features", Pipeline([("rf", RandomForestClassifier(random_state=dp.RANDOM_STATE))]), rf_grid_current, "f1", X_train_eng, X_test_eng, y_train_eng, y_test_eng)
    rows.append(row)

    progress.progress(1.0, text="Robustness checks — done ✅")

    return pd.DataFrame(rows)


def plot_robustness_roc_auc(df):
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(df))))
    labels = df["Check"] + " — " + df["Scoring"]
    colors = ["#c44e52" if v < 0.5 else "#4c72b0" for v in df["ROC-AUC"]]
    ax.barh(labels, df["ROC-AUC"], color=colors)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Chance level (0.500)")
    ax.set_xlabel("ROC-AUC")
    ax.set_title("ROC-AUC Across All Robustness Checks")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def get_feature_selection_result(X, y):
    """ANOVA top-10 feature selection robustness check (report Section 5.6.5)."""
    return load_or_compute(
        "feature_selection_anova",
        lambda: fscm.run_feature_selection_check(data={"X": X, "y": y}, save_outputs=False),
    )


@st.cache_resource(show_spinner=False)
def get_pca_result(X, y):
    """PCA dimensionality-reduction robustness check (report Section 5.6.6)."""
    return load_or_compute(
        "pca_robustness",
        lambda: pcam.run_pca_check(data={"X": X, "y": y}, save_outputs=False),
    )


def train_all_models(X, y):
    all_jobs = {
        "KNN": (train_knn_basic, X, y),
        "Decision Tree": (train_dt_basic, X, y),
        "Logistic Regression": (train_lr_basic, X, y),
        "Random Forest": (train_rf_basic, X, y),
        "KNN (SMOTE)": (train_knn_smote, X, y),
        "Decision Tree (SMOTE)": (train_dt_smote, X, y),
        "Logistic Regression (SMOTE)": (train_lr_smote, X, y),
        "Random Forest (SMOTE)": (train_rf_smote, X, y),
    }
    return run_training_jobs("Training models (Basic + SMOTE)", all_jobs)


def pick_best(basic_result, smote_result):
    basic_auc = basic_result["metrics"]["ROC-AUC"]
    smote_auc = smote_result["metrics"]["ROC-AUC"]
    if abs(basic_auc - smote_auc) > 1e-9:
        return (basic_result, "Basic") if basic_auc > smote_auc else (smote_result, "SMOTE")
    basic_acc = basic_result["metrics"]["Accuracy"]
    smote_acc = smote_result["metrics"]["Accuracy"]
    return (basic_result, "Basic") if basic_acc >= smote_acc else (smote_result, "SMOTE")


with st.sidebar:

    st.markdown("<h2 style='text-align: center;'>❤️ Heart Disease Risk</h2>", unsafe_allow_html=True)
    st.divider()
    st.markdown("#### 👤 For Everyone")

    MAIN_PAGES = ["🏠 Home (Predict & Overview)", "📊 Model Comparison"]
    MORE_PAGES = [
        "🔍 EDA",
        "🧹 Preprocessing",
        "⚖️ Basic vs SMOTE",
        "🔬 Robustness & Feature Selection",
    ]

    # Bug fix: two separate st.radio widgets each keep their own selection in
    # session_state across reruns. Without this, clicking a Main Menu option
    # never "wins" once something was picked in More — the More radio still
    # remembers its last choice and silently overrides it every rerun. A
    # single tracked value plus on_change callbacks that clear the *other*
    # widget's state fixes that: whichever one you touch last is the page.
    if "current_page" not in st.session_state:
        st.session_state.current_page = MAIN_PAGES[0]

    def _on_main_pick():
        st.session_state.current_page = st.session_state.main_page_radio
        st.session_state.more_page_radio = None

    def _on_more_pick():
        if st.session_state.more_page_radio:
            st.session_state.current_page = st.session_state.more_page_radio

    main_index = MAIN_PAGES.index(st.session_state.current_page) if st.session_state.current_page in MAIN_PAGES else None
    st.radio(
        "Navigate",
        MAIN_PAGES,
        index=main_index,
        key="main_page_radio",
        on_change=_on_main_pick,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### 🔬 For Reviewers (Technical)")
    more_active = st.session_state.current_page in MORE_PAGES
    with st.expander("Full technical breakdown", expanded=more_active):
        more_index = MORE_PAGES.index(st.session_state.current_page) if more_active else None
        st.radio(
            "More pages",
            MORE_PAGES,
            index=more_index,
            key="more_page_radio",
            on_change=_on_more_pick,
            label_visibility="collapsed",
        )

    page = st.session_state.current_page

# Scroll back to the top of the content area on an actual page switch, not
# on every rerun (a form submit or dropdown change reruns the script too,
# and we don't want to yank the user's scroll position away from a result
# they just generated — only a genuine navigation change should do that).
if st.session_state.get("_last_rendered_page") != page:
    st.session_state["_last_rendered_page"] = page
    components.html(
        """
        <script>
        (function () {
            function scrollAppToTop() {
                try {
                    window.parent.scrollTo(0, 0);
                    var selectors = [
                        '[data-testid="stAppViewContainer"]',
                        '[data-testid="stMain"]',
                        'section.main',
                        '.main'
                    ];
                    selectors.forEach(function (sel) {
                        var el = window.parent.document.querySelector(sel);
                        if (el) { el.scrollTop = 0; }
                    });
                } catch (e) {}
            }
            scrollAppToTop();
            setTimeout(scrollAppToTop, 60);
        })();
        </script>
        """,
        height=0,
    )


try:

    raw_df, numeric_cols, categorical_cols = load_raw_data(DEFAULT_DATA_PATH)
    pipeline_data = load_pipeline_data(DEFAULT_DATA_PATH)
except FileNotFoundError:
    st.error(
        f"⚠️ Error: Couldn't find `{DEFAULT_DATA_PATH}`. "
        "Please ensure the dataset is located in the exact same folder as your app.py file."
    )
    st.stop()

X, y = pipeline_data["X"], pipeline_data["y"]
le_target = pipeline_data["le_target"]
missing_treatment_summary = pipeline_data["missing_treatment_summary"]
target_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))

if page in ("🏠 Home (Predict & Overview)", "📊 Model Comparison", "🔬 Robustness & Feature Selection"):
    all_trained = train_all_models(X, y)

    knn_basic_data = all_trained["KNN"]
    X_test, y_test = knn_basic_data["X_test"], knn_basic_data["y_test"]
    knn_smote_data = all_trained["KNN (SMOTE)"]

    dt_basic_data = all_trained["Decision Tree"]
    dt_X_test, dt_y_test = dt_basic_data["X_test"], dt_basic_data["y_test"]
    dt_smote_data = all_trained["Decision Tree (SMOTE)"]

    lr_basic_data = all_trained["Logistic Regression"]
    y_test_lr = lr_basic_data["y_test"]
    lr_smote_data = all_trained["Logistic Regression (SMOTE)"]

    rf_basic_data = all_trained["Random Forest"]
    rf_X_test, rf_y_test = rf_basic_data["X_test"], rf_basic_data["y_test"]
    rf_smote_data = all_trained["Random Forest (SMOTE)"]

    basic_results = {
        "KNN": knn_basic_data["result"],
        "Logistic Regression": lr_basic_data["result"],
        "Random Forest": rf_basic_data["result"],
        "Decision Tree": dt_basic_data["result"],
    }
    smote_results = {
        "KNN": knn_smote_data["result"],
        "Logistic Regression": lr_smote_data["result"],
        "Random Forest": rf_smote_data["result"],
        "Decision Tree": dt_smote_data["result"],
    }

    best_results = {}
    best_pipeline_used = {}
    for name in basic_results:
        result, used = pick_best(basic_results[name], smote_results[name])
        best_results[name] = result
        best_pipeline_used[name] = used

    best_metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    best_df = pd.DataFrame([
        {**res["metrics"], "Model": name, "Pipeline": best_pipeline_used[name]}
        for name, res in best_results.items()
    ])

    all_results = best_results
    all_results_df = best_df


@st.cache_resource(show_spinner="Prefetching EDA Plots (One-time setup)...")
def prefetch_eda_plots(df, num_cols, cat_cols):
    return {
        "class_dist": dv.plot_class_distribution(df),
        "num_dist": dv.plot_numeric_distributions(df, num_cols),
        "qq_plots": dv.plot_qq_plots(df, num_cols),
        "cat_dist": dv.plot_categorical_distributions(df, cat_cols),
        "corr_heat": dv.plot_correlation_heatmap(df, num_cols),
        "outliers": dv.plot_outliers_boxplot(df, num_cols),
        "num_by_target": dv.plot_numeric_by_target(df, num_cols),
        "cat_rate": dv.plot_categorical_rate_by_target(df, cat_cols),
        "cat_counts": dv.plot_categorical_counts_by_target(df, cat_cols),
        "cat_pct": dv.plot_categorical_percentage_by_target(df, cat_cols)
    }

@st.cache_resource(show_spinner="Prefetching Data Stats...")
def prefetch_stats(df, num_cols, cat_cols, X_df, y_ser):
    outlier_df = dv.compute_outlier_counts(df, num_cols)
    table_numeric, table_categorical, fig_assoc = dv.test_target_associations(df, num_cols, cat_cols)
    mcar_df, fig_mcar = dv.test_alcohol_missingness_mcar(df, num_cols, cat_cols)
    fig_corr, _ = dv.plot_target_correlation_heatmap(pipeline_data["df"], X_df.columns.tolist())
    anova_df = dv.compute_anova_scores(X_df, y_ser)
    chi2_df = dv.compute_chi2_scores(X_df, y_ser)
    return outlier_df, table_numeric, table_categorical, fig_assoc, mcar_df, fig_mcar, fig_corr, anova_df, chi2_df


if page == "🏠 Home (Predict & Overview)":
    st.title("❤️ Heart Disease Risk Dashboard")
    st.html(PULSE_DIVIDER_SVG)

    st.header("🩺 Live Risk Predictor")

    model_choice = st.selectbox("Select Diagnostic Model:", list(all_results.keys()))
    best_model = all_results[model_choice]["best_model"]

    with st.form("predict_form"):
        st.subheader("Patient Vitals & Diagnostics")
        numeric_inputs = {}
        cols = st.columns(3)
        for i, col in enumerate(numeric_cols):
            default = float(raw_df[col].median())
            numeric_inputs[col] = cols[i % 3].number_input(col, value=default)

        st.divider()

        st.subheader("Patient History & Lifestyle")
        categorical_inputs = {}
        cols2 = st.columns(3)
        for i, col in enumerate(categorical_cols):
            if col in dp.ORDINAL_MAPS:
                options = list(dp.ORDINAL_MAPS[col].keys())
            else:
                options = sorted(raw_df[col].dropna().unique().tolist())
            categorical_inputs[col] = cols2[i % 3].selectbox(col, options)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
        with submit_col2:
            submitted = st.form_submit_button("Generate Prediction", width="stretch")

    if submitted:
        raw_input = {**numeric_inputs, **categorical_inputs}
        row = dp.build_single_row_features(raw_input, categorical_cols, X.columns.tolist())

        pred = int(best_model.predict(row)[0])
        prob_disease = float(best_model.predict_proba(row)[0, 1])
        label = le_target.inverse_transform([pred])[0]

        st.divider()
        st.subheader("Prediction Result")


        if pred == 1:
            st.error(f"### ⚠️ Prediction: {label}")
        else:
            st.success(f"### ✅ Prediction: {label}")


        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_disease * 100,
            number={'suffix': "%", 'font': {'size': 28, 'color': '#333'}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Calculated Risk Level", 'font': {'size': 18, 'color': '#555'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0.5)", 'thickness': 0.25},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "#d4edda"},
                    {'range': [30, 70], 'color': "#ffeeba"},
                    {'range': [70, 100], 'color': "#f8d7da"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': prob_disease * 100
                }
            }
        ))


        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))


        g_col1, g_col2, g_col3 = st.columns([1, 2, 1])
        with g_col2:
            st.plotly_chart(fig_gauge, width="stretch")

        st.caption(f"Powered by {model_choice} ({best_pipeline_used[model_choice]} pipeline)")

    st.divider()
    st.subheader("🧭 Explore Further")
    explore_col1, explore_col2 = st.columns(2)
    with explore_col1:
        with st.container(border=True):
            st.markdown("**🔍 See the Dataset**")
            st.caption(
                "10,000 patient records, class balance, missing-value patterns, correlations, "
                "and sample rows — the full exploratory analysis behind this predictor."
            )
            st.markdown("Open **🔍 EDA** from the sidebar under *For Reviewers (Technical)*.")
    with explore_col2:
        with st.container(border=True):
            st.markdown("**📊 See the Models**")
            st.caption(
                "Full metrics for all 4 algorithms, and why Random Forest is the report's "
                "recommended pick even though KNN scores a marginally higher ROC-AUC."
            )
            st.markdown("Open **📊 Model Comparison** from the sidebar under *For Everyone*.")


elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.caption(
        "Raw data only — before imputation, encoding, or modelling. "
        "Use the snapshot, then open a category tab."
    )

    eda_figs = prefetch_eda_plots(raw_df, numeric_cols, categorical_cols)
    outlier_df, table_numeric, table_categorical, fig_assoc, mcar_df, fig_mcar, fig_corr, anova_df, chi2_df = prefetch_stats(raw_df, numeric_cols, categorical_cols, X, y)

    st.subheader(":material/database: Dataset snapshot")
    st.caption(
        "10,000 patients and 21 columns (9 numeric, 11 categorical, plus Heart Disease Status)."
    )
    st.html(
        """
        <style>
        .st-key-snap_records, .st-key-snap_attrs, .st-key-snap_numeric, .st-key-snap_cat,
        .st-key-snap_balance, .st-key-snap_sample {
            border-radius: 14px !important;
        }
        .st-key-snap_records {
            background: linear-gradient(180deg, #e5f6f1 0%, #ffffff 55%) !important;
            border: 1px solid #9fd6c9 !important;
            border-top: 6px solid #0f766e !important;
        }
        .st-key-snap_attrs {
            background: linear-gradient(180deg, #fde8ea 0%, #ffffff 55%) !important;
            border: 1px solid #f3b4b8 !important;
            border-top: 6px solid #e11d48 !important;
        }
        .st-key-snap_numeric {
            background: linear-gradient(180deg, #e7f1f8 0%, #ffffff 55%) !important;
            border: 1px solid #a9c7de !important;
            border-top: 6px solid #457b9d !important;
        }
        .st-key-snap_cat {
            background: linear-gradient(180deg, #fff4d6 0%, #ffffff 55%) !important;
            border: 1px solid #efd48a !important;
            border-top: 6px solid #d4a017 !important;
        }
        .st-key-snap_balance, .st-key-snap_sample {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
        }
        </style>
        """
    )
    snapshot_kpis = [
        ("snap_records", "green", ":material/groups:", "Total Patient Records", f"{raw_df.shape[0]:,}"),
        ("snap_attrs", "red", ":material/view_column:", "Total Attributes", f"{raw_df.shape[1]}"),
        ("snap_numeric", "blue", ":material/pin:", "Numeric Features", f"{len(numeric_cols)}"),
        ("snap_cat", "orange", ":material/category:", "Categorical Features", f"{len(categorical_cols)}"),
    ]
    for col, (key, color, icon, label, value) in zip(st.columns(4), snapshot_kpis):
        with col:
            with st.container(border=True, key=key):
                st.metric(f"{icon} {label}", f":{color}[**{value}**]")

    target_counts = raw_df[dp.TARGET_COL].value_counts()
    no_n = int(target_counts.get("No", 0))
    yes_n = int(target_counts.get("Yes", 0))
    n_total = max(no_n + yes_n, 1)

    snap_col1, snap_col2 = st.columns([1, 1.5])
    with snap_col1:
        with st.container(border=True, key="snap_balance"):
            st.markdown("**Target class balance**")
            st.caption("About 4 in 5 patients have no recorded heart disease. Accuracy will look high if a model always predicts No.")
            with st.container(horizontal=True):
                st.badge(f"No  {no_n / n_total:.0%}", color="green")
                st.badge(f"Yes  {yes_n / n_total:.0%}", color="red")
            st.pyplot(eda_figs["class_dist"])
    with snap_col2:
        with st.container(border=True, key="snap_sample"):
            st.markdown("**Sample patient records (raw data)**")
            st.caption("First 15 rows, still uncleaned: original labels, missing cells, and mixed types.")
            st.dataframe(raw_df.head(15), width="stretch", hide_index=True, height=360)

    with st.container(key="eda_category_tabs"):
        eda_tab1, eda_tab2, eda_tab3 = st.tabs([
            ":material/bar_chart: **Distributions**",
            ":material/favorite: **Target associations**",
            ":material/health_and_safety: **Data quality & outliers**",
        ])

    with eda_tab1:
        st.subheader("Feature distributions")
        st.caption("How each raw field is filled, before any cleaning.")

        with st.container(border=True):
            st.markdown("**Categorical features**")
            st.caption(
                "How evenly are lifestyle and clinical categories filled? "
                "Real clinic data is usually skewed; unusually even bars are a clue the file may be synthetic."
            )
            st.pyplot(eda_figs["cat_dist"])

        with st.container(border=True):
            st.markdown("**Numeric features**")
            st.caption(
                "What shape do Age, BMI, cholesterol, and the other numeric fields have? "
                "A flat histogram (instead of a bell curve) is unusual for real patient measurements."
            )
            st.pyplot(eda_figs["num_dist"])

    with eda_tab2:
        st.subheader("How features relate to heart disease")
        st.caption(
            "Does any predictor actually separate Yes from No? "
            "Small association statistics mean a classifier has almost nothing to learn."
        )

        with st.container(border=True):
            st.markdown("**Association tests**")
            st.caption("Point-biserial r (numeric) and Cramér's V (categorical). Values near 0 mean no useful link with the target.")
            st.pyplot(fig_assoc)
            colA, colB = st.columns(2)
            colA.markdown("**Numeric (point-biserial r)**")
            colA.dataframe(table_numeric, width="stretch")
            colB.markdown("**Categorical (Cramér's V)**")
            colB.dataframe(table_categorical, width="stretch")

        with st.container(border=True):
            st.markdown("**Numeric features split by target**")
            st.caption("If a field predicted disease, the Yes and No boxes would sit at different levels. Overlapping boxes mean the two groups look the same.")
            st.pyplot(eda_figs["num_by_target"])

        with st.container(border=True):
            st.markdown("**Disease rate by category**")
            st.caption("Share of Yes within each category. If every bar is near the overall 20% rate, that category does not change risk.")
            st.pyplot(eda_figs["cat_rate"])

    with eda_tab3:
        st.subheader("Data quality & outliers")
        st.caption("Duplicates, unusual shape, correlations, and extreme values in the raw file.")

        with st.container(border=True):
            st.markdown("**Structural quality**")
            st.caption("Duplicate rows or messy category labels would be a data problem, not a modelling problem.")
            n_dupes, unique_values_df = dv.check_data_quality(raw_df, categorical_cols)
            if n_dupes == 0:
                st.success("No duplicate records found across all 10,000 rows.")
            else:
                st.warning(f"{n_dupes} duplicate record(s) found.")
            with st.expander("Unique values per categorical / binary attribute"):
                st.dataframe(unique_values_df, width="stretch", hide_index=True)

        with st.container(border=True):
            st.markdown("**Numeric correlations**")
            st.caption("Near-zero correlations mean little redundancy — and little shared clinical structure.")
            st.pyplot(eda_figs["corr_heat"])

        with st.container(border=True):
            st.markdown("**QQ-plots versus a uniform distribution**")
            st.caption("If points sit on the diagonal, the numeric field matches a flat uniform shape, matching the histograms.")
            st.pyplot(eda_figs["qq_plots"])

        with st.container(border=True):
            st.markdown("**Outlier check (1.5× IQR)**")
            st.caption("On roughly uniform variables, whiskers span most of the range, so extreme tails are limited.")
            st.pyplot(eda_figs["outliers"])
            with st.expander("Exact outlier counts"):
                st.dataframe(outlier_df, width="stretch")

    st.info(
        "Missing-value handling, encoding, and the post-encoding ANOVA / chi-square tables are on **🧹 Preprocessing**.",
        icon=":material/arrow_forward:",
    )


elif page == "🧹 Preprocessing":
    st.title("🧹 Data Preprocessing")

    outlier_df, table_numeric, table_categorical, fig_assoc, mcar_df, fig_mcar, fig_corr, anova_df, chi2_df = prefetch_stats(raw_df, numeric_cols, categorical_cols, X, y)

    prep_tab1, prep_tab2, prep_tab3 = st.tabs([
        "❓ Missing Values",
        "🔠 Encoding",
        "✅ Feature Diagnostics"
    ])

    with prep_tab1:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader("Missing Values Found")
            df_missing, fig_missing = dv.plot_missing_values(raw_df)
            if not df_missing.empty:
                st.pyplot(fig_missing)
            else:
                st.success("No missing values found in the raw data.")
        with col_m2:
            st.subheader("Missing-Value Treatment")
            display_summary = missing_treatment_summary.copy()
            display_summary["Imputation Value"] = display_summary["Imputation Value"].astype(str)
            st.dataframe(display_summary, width="stretch")

        st.divider()
        st.subheader("MCAR Test: Alcohol Consumption")
        st.markdown("Testing if missing 'Alcohol Consumption' data relates to other variables (Missing Completely At Random).")
        st.pyplot(fig_mcar)

    with prep_tab2:
        st.subheader("Categorical Encoding Logic")
        col_e1, col_e2 = st.columns([1, 1.5])
        with col_e1:
            st.markdown(
                "**1. Ordinal Encoding (Integer Mapping)**\n"
                "- **Rule:** Applied to categories with a natural, logical ranking (e.g., Low < Medium < High) so the model understands the numerical order.\n"
                "- **Features:** `Exercise Habits`, `Stress Level`, `Sugar Consumption`, `Alcohol Consumption`\n\n"
                "**2. One-Hot Encoding (Dummy Variables)**\n"
                "- **Rule:** Applied to binary or nominal categories without a specific order to prevent the model from assuming a false hierarchy.\n"
                "- **Features (both levels kept, `drop_first=False`):** `Gender`, `Smoking`, `Family Heart Disease`, `Diabetes`, `High Blood Pressure`, `Low HDL Cholesterol`, `High LDL Cholesterol`\n\n"
                "**3. Label Encoding**\n"
                f"- **Rule:** Converts the final predicted outcome into a binary machine-readable format.\n"
                f"- **Target:** `{dp.TARGET_COL}` → `{target_mapping}`"
            )
        with col_e2:
            st.write(f"**Final Feature Matrix:** {X.shape[0]:,} rows × {X.shape[1]} columns")
            st.dataframe(X.head(10), width="stretch")

    with prep_tab3:
        st.subheader("Post-Encoding Feature Diagnostics")
        st.pyplot(fig_corr)

        col_d1, col_d2 = st.columns(2)
        col_d1.markdown("**ANOVA F-scores**")
        col_d1.dataframe(anova_df, width="stretch")
        col_d2.markdown("**Chi-Square scores**")
        col_d2.dataframe(chi2_df, width="stretch")


elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")

    metrics_view = best_df[["Model", "Pipeline"] + best_metric_cols]
    recommended_row, auc_leader_row = pick_recommended_model(best_df)

    st.subheader("🏆 Evaluation Metrics")
    card_theme = {
        "KNN": {
            "key": "compare_knn", "color": "red", "badge": "red",
            "icon": ":material/scatter_plot:",
        },
        "Logistic Regression": {
            "key": "compare_logreg", "color": "green", "badge": "green",
            "icon": ":material/show_chart:",
        },
        "Random Forest": {
            "key": "compare_rf", "color": "blue", "badge": "blue",
            "icon": ":material/forest:",
        },
        "Decision Tree": {
            "key": "compare_dt", "color": "orange", "badge": "orange",
            "icon": ":material/account_tree:",
        },
    }
    st.html(
        """
        <style>
        .st-key-compare_knn, .st-key-compare_logreg, .st-key-compare_rf, .st-key-compare_dt {
            border-radius: 14px !important;
        }
        .st-key-compare_knn {
            background: linear-gradient(180deg, #fde8ea 0%, #ffffff 42%) !important;
            border: 1px solid #f3b4b8 !important;
            border-top: 6px solid #c44e52 !important;
        }
        .st-key-compare_logreg {
            background: linear-gradient(180deg, #e5f6f1 0%, #ffffff 42%) !important;
            border: 1px solid #9fd6c9 !important;
            border-top: 6px solid #2a9d8f !important;
        }
        .st-key-compare_rf {
            background: linear-gradient(180deg, #e7f1f8 0%, #ffffff 42%) !important;
            border: 1px solid #a9c7de !important;
            border-top: 6px solid #457b9d !important;
        }
        .st-key-compare_dt {
            background: linear-gradient(180deg, #fff4d6 0%, #ffffff 42%) !important;
            border: 1px solid #efd48a !important;
            border-top: 6px solid #d4a017 !important;
        }
        .st-key-compare_knn [data-testid="stMetric"]:last-of-type,
        .st-key-compare_logreg [data-testid="stMetric"]:last-of-type,
        .st-key-compare_rf [data-testid="stMetric"]:last-of-type,
        .st-key-compare_dt [data-testid="stMetric"]:last-of-type {
            background: rgba(255,255,255,0.85);
        }
        </style>
        """
    )
    model_cols = st.columns(4)
    for col, (_, row) in zip(model_cols, best_df.iterrows()):
        theme = card_theme[row["Model"]]
        accent = theme["color"]
        with col:
            with st.container(border=True, key=theme["key"]):
                st.markdown(f"{theme['icon']} :{accent}[**{row['Model']}**]")
                with st.container(horizontal=True):
                    st.badge(row["Pipeline"], color=theme["badge"])
                    if row["Model"] == recommended_row["Model"]:
                        st.badge("Recommended", icon=":material/emoji_events:", color="orange")
                    elif row["Model"] == auc_leader_row["Model"]:
                        st.badge("Highest ROC-AUC", icon=":material/trending_up:", color="gray")
                st.metric("Accuracy", f"{row['Accuracy']:.4f}")
                st.metric("Precision", f"{row['Precision']:.4f}")
                st.metric("Recall", f"{row['Recall']:.4f}")
                st.metric("F1-Score", f"{row['F1-Score']:.4f}")
                st.metric("ROC-AUC", f"{row['ROC-AUC']:.4f}", border=True)

    st.dataframe(
        metrics_view.style
            .highlight_max(subset=best_metric_cols, color="#d4edda")
            .format({c: "{:.4f}" for c in best_metric_cols}),
        width="stretch",
        hide_index=True,
    )
    if recommended_row["Model"] != auc_leader_row["Model"]:
        st.success(
            f"🏆 **{recommended_row['Model']}** ({recommended_row['Pipeline']}) is the report's recommended model "
            f"(ROC-AUC **{recommended_row['ROC-AUC']:.4f}**) — chosen over the nominal ROC-AUC leader, "
            f"**{auc_leader_row['Model']}** (**{auc_leader_row['ROC-AUC']:.4f}**), because that margin is within "
            "noise while Random Forest is more robust to the class imbalance and more interpretable."
        )
    else:
        st.success(
            f"🏆 **{recommended_row['Model']}** ({recommended_row['Pipeline']}) is both the nominal ROC-AUC leader "
            f"and the report's recommended model (**{recommended_row['ROC-AUC']:.4f}**)."
        )

    acc_row = best_df.loc[best_df["Accuracy"].idxmax()]
    rec_row = best_df.loc[best_df["Recall"].idxmax()]
    f1_row = best_df.loc[best_df["F1-Score"].idxmax()]
    no_share = float((y == 0).mean())
    max_auc = float(best_df["ROC-AUC"].max())

    st.subheader("🧭 Feature Importance (Top 10)")
    with st.container(border=True):
        with st.spinner("Computing feature importance for all 4 models..."):
            imp_knn = km.get_permutation_importance(best_results["KNN"]["best_model"], X_test, y_test).head(10)
            coef_lr = lgm.get_coefficients(best_results["Logistic Regression"]["best_model"], X.columns.tolist()).head(10)
            imp_rf = rfm.get_permutation_importance(best_results["Random Forest"]["best_model"], rf_X_test, rf_y_test).head(10)
            imp_dt = dtm.get_permutation_importance(best_results["Decision Tree"]["best_model"], dt_X_test, dt_y_test).head(10)

            feature_summary = pd.DataFrame({
                "Rank": range(1, 11),
                "KNN": [f"{r.Feature} ({r.Importance:.3f})" for r in imp_knn.itertuples()],
                "Logistic Regression": [f"{r.Feature} ({r.Coefficient:+.3f})" for r in coef_lr.itertuples()],
                "Random Forest": [f"{r.Feature} ({r.Importance:.3f})" for r in imp_rf.itertuples()],
                "Decision Tree": [f"{r.Feature} ({r.Importance:.3f})" for r in imp_dt.itertuples()],
            })
        st.dataframe(feature_summary, width="stretch", hide_index=True)


elif page == "⚖️ Basic vs SMOTE":
    st.title("⚖️ Basic vs SMOTE")

    st.html(
        """
        <style>
        .st-key-knn_acc, .st-key-knn_prec, .st-key-knn_rec, .st-key-knn_f1, .st-key-knn_auc,
        .st-key-lr_acc, .st-key-lr_prec, .st-key-lr_rec, .st-key-lr_f1, .st-key-lr_auc,
        .st-key-rf_acc, .st-key-rf_prec, .st-key-rf_rec, .st-key-rf_f1, .st-key-rf_auc,
        .st-key-dt_acc, .st-key-dt_prec, .st-key-dt_rec, .st-key-dt_f1, .st-key-dt_auc {
            border-radius: 14px !important;
        }
        .st-key-knn_acc, .st-key-lr_acc, .st-key-rf_acc, .st-key-dt_acc,
        .st-key-knn_auc, .st-key-lr_auc, .st-key-rf_auc, .st-key-dt_auc {
            background: linear-gradient(180deg, #e5f6f1 0%, #ffffff 55%) !important;
            border: 1px solid #9fd6c9 !important;
            border-top: 6px solid #0f766e !important;
        }
        .st-key-knn_prec, .st-key-lr_prec, .st-key-rf_prec, .st-key-dt_prec {
            background: linear-gradient(180deg, #fde8ea 0%, #ffffff 55%) !important;
            border: 1px solid #f3b4b8 !important;
            border-top: 6px solid #e11d48 !important;
        }
        .st-key-knn_rec, .st-key-lr_rec, .st-key-rf_rec, .st-key-dt_rec {
            background: linear-gradient(180deg, #e7f1f8 0%, #ffffff 55%) !important;
            border: 1px solid #a9c7de !important;
            border-top: 6px solid #457b9d !important;
        }
        .st-key-knn_f1, .st-key-lr_f1, .st-key-rf_f1, .st-key-dt_f1 {
            background: linear-gradient(180deg, #fff4d6 0%, #ffffff 55%) !important;
            border: 1px solid #efd48a !important;
            border-top: 6px solid #d4a017 !important;
        }
        </style>
        """
    )

    with st.container(key="smote_model_tabs"):
        model_tabs = st.tabs(["K-Nearest Neighbors (KNN)", "Logistic Regression", "Random Forest", "Decision Tree"])

    all_trained = train_all_models(X, y)

    knn_basic_data = all_trained["KNN"]
    X_test, y_test = knn_basic_data["X_test"], knn_basic_data["y_test"]
    knn_smote_data = all_trained["KNN (SMOTE)"]

    dt_basic_data = all_trained["Decision Tree"]
    dt_X_test, dt_y_test = dt_basic_data["X_test"], dt_basic_data["y_test"]
    dt_smote_data = all_trained["Decision Tree (SMOTE)"]

    lr_basic_data = all_trained["Logistic Regression"]
    y_test_lr = lr_basic_data["y_test"]
    lr_smote_data = all_trained["Logistic Regression (SMOTE)"]

    rf_basic_data = all_trained["Random Forest"]
    rf_X_test, rf_y_test = rf_basic_data["X_test"], rf_basic_data["y_test"]
    rf_smote_data = all_trained["Random Forest (SMOTE)"]

    results = {"1. Basic KNN": knn_basic_data["result"], "2. SMOTE KNN": knn_smote_data["result"]}
    results_df = pd.DataFrame([r["metrics"] for r in results.values()])

    results_lr = {"1. Basic Logistic Regression": lr_basic_data["result"], "2. SMOTE Logistic Regression": lr_smote_data["result"]}
    results_df_lr = pd.DataFrame([r["metrics"] for r in results_lr.values()])

    rf_results = {"1. Basic Random Forest": rf_basic_data["result"], "2. SMOTE Random Forest": rf_smote_data["result"]}
    rf_results_df = pd.DataFrame([r["metrics"] for r in rf_results.values()])

    dt_results = {"1. Basic Decision Tree": dt_basic_data["result"], "2. SMOTE Decision Tree": dt_smote_data["result"]}
    dt_results_df = pd.DataFrame([r["metrics"] for r in dt_results.values()])


    with model_tabs[0]:
        st.header("K-Nearest Neighbors (KNN)")

        st.subheader("📊 Baseline Metrics (& Impact of SMOTE)")

        basic = results_df.iloc[0]
        smote = results_df.iloc[1]

        def get_delta(metric):
            return float(smote[metric] - basic[metric])

        render_smote_delta_kpis("knn", basic, get_delta)

        st.divider()


        st.subheader("📈 Visual Evaluation")


        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📊 Metric Comparison", "🟦 Confusion Matrices", "📉 ROC Curves"])

        with chart_tab1:
            st.pyplot(km.plot_metric_comparison(results_df))

        with chart_tab2:
            st.pyplot(km.plot_confusion_matrices(results))

        with chart_tab3:
            roc_col1, roc_col2, roc_col3 = st.columns([1, 2, 1])
            with roc_col2:
                st.pyplot(km.plot_roc_curves(results, y_test))

        st.divider()


        st.subheader("🔍 Class-by-Class Breakdown")

        col_basic, col_smote = st.columns(2)
        models_list = list(results.keys())


        with col_basic:
            res_basic = results[models_list[0]]
            st.markdown(f"### 🔵 {models_list[0]}")
            st.info(f"**🎯 Overall Accuracy:** {res_basic['metrics']['Accuracy']:.2%} &nbsp; | &nbsp; **📈 ROC-AUC:** {res_basic['metrics']['ROC-AUC']:.2%}")

            report_basic = km.classification_report(y_test, res_basic["y_pred"], output_dict=True, zero_division=0)
            df_rep_basic = pd.DataFrame(report_basic).transpose()
            df_rep_basic.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'}, inplace=True)
            df_rep_basic = df_rep_basic.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')

            st.dataframe(df_rep_basic.style.background_gradient(cmap='Blues').format("{:.3f}"), width="stretch")


            with st.expander("⚙️ View Basic Hyperparameters"):
                st.json(res_basic['metrics']['Best Params'])


        with col_smote:
            res_smote = results[models_list[1]]
            st.markdown(f"### 🟢 {models_list[1]}")
            st.success(f"**🎯 Overall Accuracy:** {res_smote['metrics']['Accuracy']:.2%} &nbsp; | &nbsp; **📈 ROC-AUC:** {res_smote['metrics']['ROC-AUC']:.2%}")

            report_smote = km.classification_report(y_test, res_smote["y_pred"], output_dict=True, zero_division=0)
            df_rep_smote = pd.DataFrame(report_smote).transpose()
            df_rep_smote.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'}, inplace=True)
            df_rep_smote = df_rep_smote.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')

            st.dataframe(df_rep_smote.style.background_gradient(cmap='Greens').format("{:.3f}"), width="stretch")


            with st.expander("⚙️ View SMOTE Hyperparameters"):
                st.json(res_smote['metrics']['Best Params'])


        st.divider()
        st.subheader("🧭 Feature Importance (Permutation)")

        perm_tab_basic, perm_tab_smote = st.tabs([f"🔵 {models_list[0]}", f"🟢 {models_list[1]}"])

        with perm_tab_basic:
            imp_df_basic = km.get_permutation_importance(res_basic["best_model"], X_test, y_test)
            perm_col1, perm_col2 = st.columns([1.3, 1])
            with perm_col1:
                st.pyplot(km.plot_permutation_importance(imp_df_basic, "Basic KNN -- Top 15 Features"))
            with perm_col2:
                st.dataframe(
                    imp_df_basic[["Rank", "Feature", "Importance"]].style.format({"Importance": "{:.4f}"}),
                    width="stretch",
                    hide_index=True,
                )

        with perm_tab_smote:
            imp_df_smote = km.get_permutation_importance(res_smote["best_model"], X_test, y_test)
            perm_col3, perm_col4 = st.columns([1.3, 1])
            with perm_col3:
                st.pyplot(km.plot_permutation_importance(imp_df_smote, "SMOTE KNN -- Top 15 Features"))
            with perm_col4:
                st.dataframe(
                    imp_df_smote[["Rank", "Feature", "Importance"]].style.format({"Importance": "{:.4f}"}),
                    width="stretch",
                    hide_index=True,
                )


    with model_tabs[1]:
        st.header("Logistic Regression")

        lr_basic_row = results_df_lr.iloc[0]
        lr_smote_row = results_df_lr.iloc[1]

        st.subheader("📊 High-Level Metrics Impact (SMOTE vs. Basic)")

        def get_delta_lr(metric):
            return float(lr_smote_row[metric] - lr_basic_row[metric])

        render_smote_delta_kpis("lr", lr_smote_row, get_delta_lr)

        st.divider()


        st.subheader("📈 Visual Evaluation")

        lr_chart_tab1, lr_chart_tab2, lr_chart_tab3 = st.tabs(["📊 Metric Comparison", "🟦 Confusion Matrices", "📉 ROC Curves"])

        with lr_chart_tab1:
            st.pyplot(lgm.plot_metric_comparison(results_df_lr))

        with lr_chart_tab2:
            st.pyplot(lgm.plot_confusion_matrices(results_lr))

        with lr_chart_tab3:
            lr_roc_col1, lr_roc_col2, lr_roc_col3 = st.columns([1, 2, 1])
            with lr_roc_col2:
                st.pyplot(lgm.plot_roc_curves(results_lr, y_test_lr))

        st.divider()


        st.subheader("🔍 Class-by-Class Breakdown")

        col_lr_basic, col_lr_smote = st.columns(2)
        lr_models_list = list(results_lr.keys())

        with col_lr_basic:
            res_lr_basic = results_lr[lr_models_list[0]]
            st.markdown(f"### 🔵 {lr_models_list[0]}")
            st.info(f"**🎯 Overall Accuracy:** {res_lr_basic['metrics']['Accuracy']:.2%} &nbsp; | &nbsp; **📈 ROC-AUC:** {res_lr_basic['metrics']['ROC-AUC']:.2%}")

            report_lr_basic = lgm.classification_report(y_test_lr, res_lr_basic["y_pred"], output_dict=True, zero_division=0)
            df_rep_lr_basic = pd.DataFrame(report_lr_basic).transpose()
            df_rep_lr_basic.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'}, inplace=True)
            df_rep_lr_basic = df_rep_lr_basic.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')

            st.dataframe(df_rep_lr_basic.style.background_gradient(cmap='Blues').format("{:.3f}"), width="stretch")

            with st.expander("⚙️ View Basic Hyperparameters"):
                st.json(res_lr_basic['metrics']['Best Params'])

        with col_lr_smote:
            res_lr_smote = results_lr[lr_models_list[1]]
            st.markdown(f"### 🟢 {lr_models_list[1]}")
            st.success(f"**🎯 Overall Accuracy:** {res_lr_smote['metrics']['Accuracy']:.2%} &nbsp; | &nbsp; **📈 ROC-AUC:** {res_lr_smote['metrics']['ROC-AUC']:.2%}")

            report_lr_smote = lgm.classification_report(y_test_lr, res_lr_smote["y_pred"], output_dict=True, zero_division=0)
            df_rep_lr_smote = pd.DataFrame(report_lr_smote).transpose()
            df_rep_lr_smote.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'}, inplace=True)
            df_rep_lr_smote = df_rep_lr_smote.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')

            st.dataframe(df_rep_lr_smote.style.background_gradient(cmap='Greens').format("{:.3f}"), width="stretch")

            with st.expander("⚙️ View SMOTE Hyperparameters"):
                st.json(res_lr_smote['metrics']['Best Params'])

        st.divider()


        st.subheader("🧭 Coefficient Interpretability")
        st.write(
            "Since features are standardized before fitting, each coefficient's magnitude reflects "
            "how strongly that feature moves predicted risk -- positive pushes toward Disease, negative toward No Disease."
        )

        coef_tab_basic, coef_tab_smote = st.tabs([f"🔵 {lr_models_list[0]}", f"🟢 {lr_models_list[1]}"])

        with coef_tab_basic:
            coef_df_basic = lgm.get_coefficients(res_lr_basic["best_model"], X.columns.tolist())
            coef_col1, coef_col2 = st.columns([1.3, 1])
            with coef_col1:
                st.pyplot(lgm.plot_coefficients(coef_df_basic, "Basic Logistic Regression -- Feature Influence"))
            with coef_col2:
                st.dataframe(
                    coef_df_basic[["Rank", "Feature", "Coefficient", "Effect"]],
                    width="stretch",
                    hide_index=True,
                )

        with coef_tab_smote:
            coef_df_smote = lgm.get_coefficients(res_lr_smote["best_model"], X.columns.tolist())
            coef_col3, coef_col4 = st.columns([1.3, 1])
            with coef_col3:
                st.pyplot(lgm.plot_coefficients(coef_df_smote, "SMOTE Logistic Regression -- Feature Influence"))
            with coef_col4:
                st.dataframe(
                    coef_df_smote[["Rank", "Feature", "Coefficient", "Effect"]],
                    width="stretch",
                    hide_index=True,
                )


    with model_tabs[2]:
        st.header("Random Forest")

        rf_basic = rf_results_df.iloc[0]
        rf_smote = rf_results_df.iloc[1]

        st.subheader("📊 High-Level Metrics Impact : SMOTE vs. Basic")

        def get_rf_delta(metric):
            return float(rf_smote[metric] - rf_basic[metric])

        render_smote_delta_kpis("rf", rf_smote, get_rf_delta)

        st.divider()


        st.subheader("📈 Visual Evaluation")

        rf_tab1, rf_tab2, rf_tab3 = st.tabs(["📊 Metric Comparison", "🟦 Confusion Matrices", "📉 ROC Curves"])

        with rf_tab1:
            st.pyplot(rfm.plot_metric_comparison(rf_results_df))

        with rf_tab2:
            st.pyplot(rfm.plot_confusion_matrices(rf_results))

        with rf_tab3:
            rf_roc1, rf_roc2, rf_roc3 = st.columns([1, 2, 1])
            with rf_roc2:
                st.pyplot(rfm.plot_roc_curves(rf_results, rf_y_test))

        st.divider()


        st.subheader("🔍 Class-by-Class Breakdown")

        rf_col_basic, rf_col_smote = st.columns(2)
        rf_models_list = list(rf_results.keys())


        with rf_col_basic:
            rf_res_basic = rf_results[rf_models_list[0]]
            st.markdown(f"### 🔵 {rf_models_list[0]}")
            st.info(f"**🎯 Overall Accuracy:** {rf_res_basic['metrics']['Accuracy']:.2%} &nbsp; | &nbsp; **📈 ROC-AUC:** {rf_res_basic['metrics']['ROC-AUC']:.2%}")

            rf_report_basic = rfm.classification_report(rf_y_test, rf_res_basic["y_pred"], output_dict=True, zero_division=0)
            rf_df_rep_basic = pd.DataFrame(rf_report_basic).transpose()
            rf_df_rep_basic.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'}, inplace=True)
            rf_df_rep_basic = rf_df_rep_basic.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')

            st.dataframe(rf_df_rep_basic.style.background_gradient(cmap='Blues').format("{:.3f}"), width="stretch")

            with st.expander("⚙️ View Basic Hyperparameters"):
                st.json(rf_res_basic['metrics']['Best Params'])


        with rf_col_smote:
            rf_res_smote = rf_results[rf_models_list[1]]
            st.markdown(f"### 🟢 {rf_models_list[1]}")
            st.success(f"**🎯 Overall Accuracy:** {rf_res_smote['metrics']['Accuracy']:.2%} &nbsp; | &nbsp; **📈 ROC-AUC:** {rf_res_smote['metrics']['ROC-AUC']:.2%}")

            rf_report_smote = rfm.classification_report(rf_y_test, rf_res_smote["y_pred"], output_dict=True, zero_division=0)
            rf_df_rep_smote = pd.DataFrame(rf_report_smote).transpose()
            rf_df_rep_smote.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'}, inplace=True)
            rf_df_rep_smote = rf_df_rep_smote.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')

            st.dataframe(rf_df_rep_smote.style.background_gradient(cmap='Greens').format("{:.3f}"), width="stretch")

            with st.expander("⚙️ View SMOTE Hyperparameters"):
                st.json(rf_res_smote['metrics']['Best Params'])


        st.divider()
        st.subheader("🧭 Permutation Feature Importance")

        rf_perm_basic, rf_perm_smote = st.tabs([f"🔵 {rf_models_list[0]}", f"🟢 {rf_models_list[1]}"])

        with rf_perm_basic:
            rf_imp_basic = rfm.get_permutation_importance(rf_res_basic["best_model"], rf_X_test, rf_y_test)
            rf_perm_col1, rf_perm_col2 = st.columns([1.3, 1])
            with rf_perm_col1:
                st.pyplot(rfm.plot_permutation_importance(rf_imp_basic, "Basic Random Forest -- All Features"))
            with rf_perm_col2:
                st.dataframe(
                    rf_imp_basic[["Rank", "Feature", "Importance"]].style.format({"Importance": "{:.4f}"}),
                    width="stretch",
                    hide_index=True,
                )

        with rf_perm_smote:
            rf_imp_smote = rfm.get_permutation_importance(rf_res_smote["best_model"], rf_X_test, rf_y_test)
            rf_perm_col3, rf_perm_col4 = st.columns([1.3, 1])
            with rf_perm_col3:
                st.pyplot(rfm.plot_permutation_importance(rf_imp_smote, "SMOTE Random Forest -- All Features"))
            with rf_perm_col4:
                st.dataframe(
                    rf_imp_smote[["Rank", "Feature", "Importance"]].style.format({"Importance": "{:.4f}"}),
                    width="stretch",
                    hide_index=True,
                )


    with model_tabs[3]:
        st.header("Decision Tree")

        dt_basic = dt_results_df.iloc[0]
        dt_smote = dt_results_df.iloc[1]

        st.subheader("📊 High-Level Metrics Impact : SMOTE vs. Basic")

        def get_dt_delta(metric):
            return float(dt_smote[metric] - dt_basic[metric])

        render_smote_delta_kpis("dt", dt_smote, get_dt_delta)

        st.divider()


        st.subheader("📈 Visual Evaluation")

        dt_tab1, dt_tab2, dt_tab3 = st.tabs(["📊 Metric Comparison", "🟦 Confusion Matrices", "📉 ROC Curves"])

        with dt_tab1:
            st.pyplot(dtm.plot_metric_comparison(dt_results_df))

        with dt_tab2:
            st.pyplot(dtm.plot_confusion_matrices(dt_results))

        with dt_tab3:
            dt_roc1, dt_roc2, dt_roc3 = st.columns([1, 2, 1])
            with dt_roc2:
                st.pyplot(dtm.plot_roc_curves(dt_results, dt_y_test))

        st.divider()


        st.subheader("🔍 Class-by-Class Breakdown")

        dt_col_basic, dt_col_smote = st.columns(2)
        dt_models_list = list(dt_results.keys())


        with dt_col_basic:
            dt_res_basic = dt_results[dt_models_list[0]]
            st.markdown(f"### 🔵 {dt_models_list[0]}")
            st.info(f"**🎯 Overall Accuracy:** {dt_res_basic['metrics']['Accuracy']:.2%} &nbsp; | &nbsp; **📈 ROC-AUC:** {dt_res_basic['metrics']['ROC-AUC']:.2%}")

            dt_report_basic = dtm.classification_report(dt_y_test, dt_res_basic["y_pred"], output_dict=True, zero_division=0)
            dt_df_rep_basic = pd.DataFrame(dt_report_basic).transpose()
            dt_df_rep_basic.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'}, inplace=True)
            dt_df_rep_basic = dt_df_rep_basic.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')

            st.dataframe(dt_df_rep_basic.style.background_gradient(cmap='Blues').format("{:.3f}"), width="stretch")

            with st.expander("⚙️ View Basic Hyperparameters"):
                st.json(dt_res_basic['metrics']['Best Params'])


        with dt_col_smote:
            dt_res_smote = dt_results[dt_models_list[1]]
            st.markdown(f"### 🟢 {dt_models_list[1]}")
            st.success(f"**🎯 Overall Accuracy:** {dt_res_smote['metrics']['Accuracy']:.2%} &nbsp; | &nbsp; **📈 ROC-AUC:** {dt_res_smote['metrics']['ROC-AUC']:.2%}")

            dt_report_smote = dtm.classification_report(dt_y_test, dt_res_smote["y_pred"], output_dict=True, zero_division=0)
            dt_df_rep_smote = pd.DataFrame(dt_report_smote).transpose()
            dt_df_rep_smote.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'}, inplace=True)
            dt_df_rep_smote = dt_df_rep_smote.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')

            st.dataframe(dt_df_rep_smote.style.background_gradient(cmap='Greens').format("{:.3f}"), width="stretch")

            with st.expander("⚙️ View SMOTE Hyperparameters"):
                st.json(dt_res_smote['metrics']['Best Params'])


        st.divider()
        st.subheader("🧭 Permutation Feature Importance")

        dt_perm_basic, dt_perm_smote = st.tabs([f"🔵 {dt_models_list[0]}", f"🟢 {dt_models_list[1]}"])

        with dt_perm_basic:
            dt_imp_basic = dtm.get_permutation_importance(dt_res_basic["best_model"], dt_X_test, dt_y_test)
            dt_perm_col1, dt_perm_col2 = st.columns([1.3, 1])
            with dt_perm_col1:
                st.pyplot(dtm.plot_permutation_importance(dt_imp_basic, "Basic Decision Tree -- All Features"))
            with dt_perm_col2:
                st.dataframe(
                    dt_imp_basic[["Rank", "Feature", "Importance"]].style.format({"Importance": "{:.4f}"}),
                    width="stretch",
                    hide_index=True,
                )

        with dt_perm_smote:
            dt_imp_smote = dtm.get_permutation_importance(dt_res_smote["best_model"], dt_X_test, dt_y_test)
            dt_perm_col3, dt_perm_col4 = st.columns([1.3, 1])
            with dt_perm_col3:
                st.pyplot(dtm.plot_permutation_importance(dt_imp_smote, "SMOTE Decision Tree -- All Features"))
            with dt_perm_col4:
                st.dataframe(
                    dt_imp_smote[["Rank", "Feature", "Importance"]].style.format({"Importance": "{:.4f}"}),
                    width="stretch",
                    hide_index=True,
                )

        st.divider()
        st.subheader("🌳 Tree Structure (Basic Decision Tree)")
        with st.expander("Show the fitted tree (first 4 of 8 levels)", expanded=False):
            st.caption(
                "The full tree is 8 levels / 215 nodes and too wide to read as one figure, "
                "so only the top 4 levels are drawn here — matching Figure 5.9 in the report."
            )
            st.pyplot(dtm.plot_fitted_tree(dt_res_basic["best_model"], dt_X_test.columns))


elif page == "🔬 Robustness & Feature Selection":
    st.title("🔬 Robustness & Feature Selection")
    st.caption(
        "Section 5.6 of the report: checks whether the near-chance ROC-AUC (≈0.49–0.52) "
        "seen in every model is a tuning artefact, or a real limit of the data."
    )

    full_rf_auc = all_results["Random Forest"]["metrics"]["ROC-AUC"]

    st.html(
        """
        <style>
        .st-key-fs_acc, .st-key-fs_prec, .st-key-fs_rec, .st-key-fs_f1, .st-key-fs_auc,
        .st-key-pca_orig, .st-key-pca_comp, .st-key-pca_auc,
        .st-key-fs_anova_chart, .st-key-fs_anova_table, .st-key-fs_cm, .st-key-fs_roc,
        .st-key-rob_chart, .st-key-rob_table, .st-key-pca_var, .st-key-pca_cm, .st-key-pca_roc {
            border-radius: 14px !important;
        }
        .st-key-fs_acc, .st-key-pca_orig {
            background: linear-gradient(180deg, #e5f6f1 0%, #ffffff 55%) !important;
            border: 1px solid #9fd6c9 !important;
            border-top: 6px solid #0f766e !important;
        }
        .st-key-fs_prec {
            background: linear-gradient(180deg, #fde8ea 0%, #ffffff 55%) !important;
            border: 1px solid #f3b4b8 !important;
            border-top: 6px solid #e11d48 !important;
        }
        .st-key-fs_rec, .st-key-pca_comp {
            background: linear-gradient(180deg, #e7f1f8 0%, #ffffff 55%) !important;
            border: 1px solid #a9c7de !important;
            border-top: 6px solid #457b9d !important;
        }
        .st-key-fs_f1 {
            background: linear-gradient(180deg, #fff4d6 0%, #ffffff 55%) !important;
            border: 1px solid #efd48a !important;
            border-top: 6px solid #d4a017 !important;
        }
        .st-key-fs_auc, .st-key-pca_auc {
            background: linear-gradient(180deg, #e5f6f1 0%, #ffffff 55%) !important;
            border: 1px solid #9fd6c9 !important;
            border-top: 6px solid #0f766e !important;
        }
        .st-key-fs_anova_chart, .st-key-fs_anova_table, .st-key-fs_cm, .st-key-fs_roc,
        .st-key-rob_chart, .st-key-rob_table, .st-key-pca_var, .st-key-pca_cm, .st-key-pca_roc {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
        }
        .st-key-fs_kpi_row [data-testid="stHorizontalBlock"],
        .st-key-pca_kpi_row [data-testid="stHorizontalBlock"] {
            align-items: stretch !important;
        }
        .st-key-fs_kpi_row [data-testid="stColumn"],
        .st-key-pca_kpi_row [data-testid="stColumn"] {
            display: flex !important;
            flex-direction: column !important;
        }
        .st-key-fs_kpi_row [data-testid="stColumn"] > div,
        .st-key-pca_kpi_row [data-testid="stColumn"] > div,
        .st-key-fs_acc, .st-key-fs_prec, .st-key-fs_rec, .st-key-fs_f1, .st-key-fs_auc,
        .st-key-pca_orig, .st-key-pca_comp, .st-key-pca_auc {
            height: 100% !important;
            flex: 1 1 auto !important;
        }
        .st-key-fs_kpi_row [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-pca_kpi_row [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-fs_kpi_row [data-testid="stVerticalBlockBorderWrapper"] > div,
        .st-key-pca_kpi_row [data-testid="stVerticalBlockBorderWrapper"] > div {
            height: 100% !important;
            min-height: 11.5rem !important;
        }
        .st-key-fs_kpi_row [data-testid="stMetric"],
        .st-key-pca_kpi_row [data-testid="stMetric"] {
            height: 100% !important;
            box-sizing: border-box !important;
        }
        </style>
        """
    )

    with st.container(key="rob_category_tabs"):
        rob_tab1, rob_tab2, rob_tab3 = st.tabs([
            "⚙️ Scoring, Threshold & Grid Checks",
            "🎯 Feature Selection (ANOVA Top-10)",
            "🧬 Dimensionality Reduction (PCA)",
        ])

    with rob_tab1:
        robustness_df = run_robustness_checks(X, y)
        metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

        with st.container(border=True, key="rob_chart"):
            st.markdown("**ROC-AUC across all checks**")
            st.caption("If every check stays near 0.50, the weak result is not from one scoring choice or threshold.")
            st.pyplot(plot_robustness_roc_auc(robustness_df))

        with st.container(border=True, key="rob_table"):
            st.markdown("**Full results**")
            st.dataframe(
                robustness_df.style.format({c: "{:.4f}" for c in metric_cols}),
                width="stretch",
                hide_index=True,
            )

    with rob_tab2:
        st.subheader("Random Forest retrained on the top 10 ANOVA-ranked features")
        st.caption("Report Section 5.6.5 / Table 5.5 — does trimming to the strongest-scoring predictors change anything?")
        fs_result = with_spinner("Retraining Random Forest on the top 10 ANOVA features...", get_feature_selection_result, X, y)

        fs_metrics = fs_result["metrics"]
        auc_delta = fs_metrics["ROC-AUC"] - full_rf_auc
        fs_kpis = [
            ("fs_acc", "green", ":material/verified:", "Accuracy", f"{fs_metrics['Accuracy']:.4f}", None),
            ("fs_prec", "red", ":material/filter_alt:", "Precision", f"{fs_metrics['Precision']:.4f}", None),
            ("fs_rec", "blue", ":material/track_changes:", "Recall", f"{fs_metrics['Recall']:.4f}", None),
            ("fs_f1", "orange", ":material/balance:", "F1-Score", f"{fs_metrics['F1-Score']:.4f}", None),
            ("fs_auc", "green", ":material/monitoring:", "ROC-AUC", f"{fs_metrics['ROC-AUC']:.4f}", f"{auc_delta:+.4f} vs full-feature RF"),
        ]
        with st.container(key="fs_kpi_row"):
            for col, (key, color, icon, label, value, delta) in zip(st.columns(5), fs_kpis):
                with col:
                    with st.container(border=True, key=key):
                        st.metric(f"{icon} {label}", f":{color}[**{value}**]", delta)

        fs_col1, fs_col2 = st.columns([1.3, 1])
        with fs_col1:
            with st.container(border=True, key="fs_anova_chart"):
                st.markdown("**Top 10 features by ANOVA F-score**")
                st.pyplot(fs_result["fig_anova_scores"])
        with fs_col2:
            with st.container(border=True, key="fs_anova_table"):
                st.markdown("**Top 10 features by ANOVA F-score**")
                st.dataframe(fs_result["anova_results"].head(10), width="stretch", hide_index=True)

        fs_col3, fs_col4 = st.columns(2)
        with fs_col3:
            with st.container(border=True, key="fs_cm"):
                st.pyplot(fs_result["fig_confusion_matrix"])
        with fs_col4:
            with st.container(border=True, key="fs_roc"):
                st.pyplot(fs_result["fig_roc_curve"])

        st.info(
            f"ROC-AUC stays at **{fs_metrics['ROC-AUC']:.4f}**, essentially unchanged from the "
            f"full-feature Random Forest (**{full_rf_auc:.4f}**). Trimming to the 10 strongest-scoring "
            "predictors neither helps nor hurts — the signal simply isn't there to find."
        )

    with rob_tab3:
        st.subheader("Random Forest retrained on PCA-reduced features")
        st.caption("Report Section 5.6.6 / Figure 5.18 — is redundancy between features hiding the signal?")
        pca_result = with_spinner("Fitting PCA and retraining Random Forest on the reduced features...", get_pca_result, X, y)

        pca_auc_delta = pca_result["metrics"]["ROC-AUC"] - full_rf_auc
        pca_kpis = [
            ("pca_orig", "green", ":material/view_column:", "Original features", f"{pca_result['metrics']['Original Features']}", None),
            ("pca_comp", "blue", ":material/tune:", "Components for 95% variance", f"{pca_result['metrics']['PCA Components']}", None),
            ("pca_auc", "green", ":material/monitoring:", "ROC-AUC", f"{pca_result['metrics']['ROC-AUC']:.4f}", f"{pca_auc_delta:+.4f} vs full-feature RF"),
        ]
        with st.container(key="pca_kpi_row"):
            for col, (key, color, icon, label, value, delta) in zip(st.columns(3), pca_kpis):
                with col:
                    with st.container(border=True, key=key):
                        st.metric(f"{icon} {label}", f":{color}[**{value}**]", delta)

        with st.container(border=True, key="pca_var"):
            st.pyplot(pca_result["fig_explained_variance"])

        pca_col1, pca_col2 = st.columns(2)
        with pca_col1:
            with st.container(border=True, key="pca_cm"):
                st.pyplot(pca_result["fig_confusion_matrix"])
        with pca_col2:
            with st.container(border=True, key="pca_roc"):
                st.pyplot(pca_result["fig_roc_curve"])

        st.info(
            f"{pca_result['metrics']['PCA Components']} of {pca_result['metrics']['Original Features']} "
            "components are needed to keep 95% of the variance — little redundancy to remove. "
            f"ROC-AUC (**{pca_result['metrics']['ROC-AUC']:.4f}**) stays in the same 0.49–0.52 band as every "
            "other check, so dimensionality reduction doesn't uncover hidden signal either."
        )