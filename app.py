import hashlib
import os

import joblib
import numpy as np
import streamlit as st
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

st.set_page_config(page_title="Heart Disease Risk", layout="wide", page_icon="\u2764\ufe0f")

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
    outputs = {}
    with st.status(label, expanded=True) as status:
        for name, (func, X, y) in jobs.items():
            outputs[name] = func(X, y)
            status.write(f"✅ {name} ready")
        status.update(label=f"{label} — done", state="complete")
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
    from sklearn.model_selection import train_test_split
    X_eng = _add_interaction_terms(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=dp.RANDOM_STATE, stratify=y)
    X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(X_eng, y, test_size=0.30, random_state=dp.RANDOM_STATE, stratify=y)

    dt_grid_current = {"dt__criterion": ["gini", "entropy"], "dt__max_depth": [3, 5, 8], "dt__min_samples_leaf": [10, 25, 50]}
    dt_grid_widened = {"dt__criterion": ["gini", "entropy"], "dt__max_depth": [3, 5, 8, 12, None], "dt__min_samples_leaf": [5, 10, 25, 50], "dt__min_samples_split": [2, 10, 20], "dt__class_weight": [None, "balanced"]}
    rf_grid_current = {"rf__n_estimators": [100, 300], "rf__min_samples_leaf": [5, 10], "rf__class_weight": ["balanced"]}
    rf_grid_free_weight = {"rf__n_estimators": [100, 300], "rf__min_samples_leaf": [5, 10], "rf__class_weight": [None, "balanced"]}

    rows = []
    with st.status("Running robustness checks", expanded=True) as status:
        for scoring in ["f1", "accuracy", "roc_auc", "recall"]:
            row, _, _ = _eval_pipeline("KNN", f"scoring={scoring}", Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))]), {}, scoring, X_train, X_test, y_train, y_test)
            rows.append(row)
            row, _, _ = _eval_pipeline("Logistic Regression", f"scoring={scoring}", Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=5000, random_state=dp.RANDOM_STATE))]), {}, scoring, X_train, X_test, y_train, y_test)
            rows.append(row)
            row, _, _ = _eval_pipeline("Decision Tree", f"scoring={scoring}", Pipeline([("dt", DecisionTreeClassifier(random_state=dp.RANDOM_STATE))]), dt_grid_current, scoring, X_train, X_test, y_train, y_test)
            rows.append(row)
            row, _, _ = _eval_pipeline("Random Forest", f"scoring={scoring}", Pipeline([("rf", RandomForestClassifier(random_state=dp.RANDOM_STATE))]), rf_grid_current, scoring, X_train, X_test, y_train, y_test)
            rows.append(row)
            status.write(f"✅ scoring={scoring} done")

        row, dt_model, dt_prob = _eval_pipeline("Decision Tree", "f1 (baseline)", Pipeline([("dt", DecisionTreeClassifier(random_state=dp.RANDOM_STATE))]), dt_grid_current, "f1", X_train, X_test, y_train, y_test)
        tuned_t = _best_threshold_from_cv(dt_model, X_train, y_train)
        y_pred_t = (dt_prob >= tuned_t).astype(int)
        rows.append({
            "Check": "Decision Tree", "Scoring": f"f1 + threshold tuned (t={tuned_t:.3f})",
            "Accuracy": round(accuracy_score(y_test, y_pred_t), 4), "Precision": round(precision_score(y_test, y_pred_t, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred_t, zero_division=0), 4), "F1-Score": round(f1_score(y_test, y_pred_t, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, dt_prob), 4),
        })
        status.write("✅ threshold tuning done")

        row, _, _ = _eval_pipeline("Decision Tree", "widened grid (class_weight incl.)", Pipeline([("dt", DecisionTreeClassifier(random_state=dp.RANDOM_STATE))]), dt_grid_widened, "f1", X_train, X_test, y_train, y_test)
        rows.append(row)
        row, _, _ = _eval_pipeline("Random Forest", "class_weight=None allowed", Pipeline([("rf", RandomForestClassifier(random_state=dp.RANDOM_STATE))]), rf_grid_free_weight, "f1", X_train, X_test, y_train, y_test)
        rows.append(row)
        status.write("✅ widened grids done")

        row, _, _ = _eval_pipeline("Decision Tree", "+ interaction features", Pipeline([("dt", DecisionTreeClassifier(random_state=dp.RANDOM_STATE))]), dt_grid_current, "f1", X_train_eng, X_test_eng, y_train_eng, y_test_eng)
        rows.append(row)
        row, _, _ = _eval_pipeline("Random Forest", "+ interaction features", Pipeline([("rf", RandomForestClassifier(random_state=dp.RANDOM_STATE))]), rf_grid_current, "f1", X_train_eng, X_test_eng, y_train_eng, y_test_eng)
        rows.append(row)
        status.write("✅ feature engineering done")
        status.update(label="Robustness checks — done", state="complete")

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

    st.markdown(
        """
        <style>
        /* Hide the radio button circles */
        div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        /* Make the sidebar buttons wide, tall, and easy to click */
        div[role="radiogroup"] label {
            display: flex;
            align-items: center;
            width: 100%;
            padding: 12px 16px;
            margin-bottom: 8px;
            border-radius: 8px;
            background-color: rgba(0, 0, 0, 0.03);
            transition: all 0.2s ease-in-out;
            cursor: pointer;
            font-weight: 500;
        }
        /* Hover effect */
        div[role="radiogroup"] label:hover {
            background-color: rgba(196, 78, 82, 0.1) !important;
            color: #c44e52 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.markdown("<h2 style='text-align: center; color: #c44e52;'>❤️ Heart Disease Prediction</h2>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 🧭 Main Menu")

    main_page = st.radio(
        "Navigate",
        [
            "🏠 Home (Predict & Overview)",
            "📊 Model Comparison",
        ],
        label_visibility="collapsed"
    )

    st.divider()
    with st.expander("🔧 More"):
        extra_page = st.radio(
            "More pages",
            [
                "🔍 EDA",
                "🧹 Preprocessing",
                "⚖️ Basic vs SMOTE",
                "🔬 Robustness Checks",
            ],
            index=None,
            label_visibility="collapsed",
        )

    page = extra_page if extra_page else main_page


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

if page in ("🏠 Home (Predict & Overview)", "📊 Model Comparison"):
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
    st.markdown("Use the calculator below to assess patient risk, or scroll down to view the dataset overview.")


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
    st.header("📊 Dataset Overview")


    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patient Records", f"{raw_df.shape[0]:,}")
    c2.metric("Total Attributes", raw_df.shape[1])
    c3.metric("Numeric Features", len(numeric_cols))
    c4.metric("Categorical Features", len(categorical_cols))

    st.markdown("<br>", unsafe_allow_html=True)

    overview_col1, overview_col2 = st.columns([1, 1.5])

    with overview_col1:
        st.markdown("**Target Class Balance**")
        st.pyplot(dv.plot_class_distribution(raw_df))

    with overview_col2:
        st.markdown("**Current Top Performing Model**")
        best_name = all_results_df.loc[all_results_df["ROC-AUC"].idxmax(), "Model"]
        best_auc = all_results_df["ROC-AUC"].max()
        st.info(f"🏆 **{best_name}** currently leads with a test ROC-AUC of **{best_auc:.3f}**.")


        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📂 View Sample Patient Records (Raw Data)"):
            st.dataframe(raw_df.head(15), width="stretch")


elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Explore the underlying patterns in the raw dataset before any cleaning or encoding.")

    eda_figs = prefetch_eda_plots(raw_df, numeric_cols, categorical_cols)
    outlier_df, table_numeric, table_categorical, fig_assoc, mcar_df, fig_mcar, fig_corr, anova_df, chi2_df = prefetch_stats(raw_df, numeric_cols, categorical_cols, X, y)

    eda_tab1, eda_tab2, eda_tab3 = st.tabs([
        "📊 Distributions",
        "🎯 Target Associations",
        "🔬 Data Quality & Outliers"
    ])

    with eda_tab1:
        st.subheader("Feature & Class Distributions")

        col_c1, col_c2 = st.columns([1, 1.5])
        with col_c1:
            st.pyplot(eda_figs["class_dist"])
        with col_c2:
            st.pyplot(eda_figs["cat_dist"])

        st.divider()
        st.pyplot(eda_figs["num_dist"])

    with eda_tab2:
        st.subheader("How Features Relate to Heart Disease")

        st.pyplot(fig_assoc)

        colA, colB = st.columns(2)
        colA.markdown("**Numeric (Point-Biserial r)**")
        colA.dataframe(table_numeric, width="stretch")
        colB.markdown("**Categorical (Cramer's V)**")
        colB.dataframe(table_categorical, width="stretch")

        st.divider()
        st.pyplot(eda_figs["num_by_target"])
        st.pyplot(eda_figs["cat_rate"])

    with eda_tab3:
        st.subheader("Correlations & Normality")

        st.pyplot(eda_figs["corr_heat"])
        st.pyplot(eda_figs["qq_plots"])

        st.divider()
        st.subheader("Outlier Detection (1.5× IQR)")
        st.pyplot(eda_figs["outliers"])
        with st.expander("📂 View Exact Outlier Counts"):
            st.dataframe(outlier_df, width="stretch")


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


# Unused leftover page (not in the sidebar). Live prediction is on Home.
# `results` was never defined here; Home uses `all_results` instead.
#
# elif page == "\U0001F52E Predict":
#     st.title("Predict Heart Disease Risk")
#     st.caption("Fill in the fields, pick a model, and get a live prediction.")
#
#     model_choice = st.selectbox("Model", list(results.keys()))
#     best_model = results[model_choice]["best_model"]
#
#     with st.form("predict_form"):
#         st.subheader("Numeric")
#         numeric_inputs = {}
#         cols = st.columns(3)
#         for i, col in enumerate(numeric_cols):
#             default = float(raw_df[col].median())
#             numeric_inputs[col] = cols[i % 3].number_input(col, value=default)
#
#         st.subheader("Categorical")
#         categorical_inputs = {}
#         cols2 = st.columns(3)
#         for i, col in enumerate(categorical_cols):
#             if col in dp.ORDINAL_MAPS:
#                 options = list(dp.ORDINAL_MAPS[col].keys())
#             else:
#                 options = sorted(raw_df[col].dropna().unique().tolist())
#             categorical_inputs[col] = cols2[i % 3].selectbox(col, options)
#
#         submitted = st.form_submit_button("Predict")
#
#     if submitted:
#         raw_input = {**numeric_inputs, **categorical_inputs}
#         row = dp.build_single_row_features(raw_input, categorical_cols, X.columns.tolist())
#
#         pred = int(best_model.predict(row)[0])
#         prob_disease = float(best_model.predict_proba(row)[0, 1])
#         label = le_target.inverse_transform([pred])[0]
#
#         st.subheader("Result")
#         if pred == 1:
#             st.error(f"Prediction: **{label}** \u2014 probability of heart disease: {prob_disease:.1%}")
#         else:
#             st.success(f"Prediction: **{label}** \u2014 probability of heart disease: {prob_disease:.1%}")
#         st.progress(min(max(prob_disease, 0.0), 1.0))
#         st.caption(
#             f"Model: {model_choice} \u2014 Best Params: `{results[model_choice]['metrics']['Best Params']}`"
#         )


elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")

    st.subheader("🏆 Evaluation Metrics")
    st.dataframe(
        best_df[["Model", "Pipeline"] + best_metric_cols].style
            .highlight_max(subset=best_metric_cols, color="#d4edda")
            .format({c: "{:.4f}" for c in best_metric_cols}),
        width="stretch",
        hide_index=True,
    )
    top_row = best_df.loc[best_df["ROC-AUC"].idxmax()]
    st.success(f"🏆 **{top_row['Model']}** ({top_row['Pipeline']}) currently leads on test ROC-AUC (**{top_row['ROC-AUC']:.4f}**).")

    st.divider()

    st.subheader("🧭 Feature Importance (Top 10)")

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


        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{basic['Accuracy']:.4f}", f"{get_delta('Accuracy'):.4f}")
        c2.metric("Precision", f"{basic['Precision']:.4f}", f"{get_delta('Precision'):.4f}")
        c3.metric("Recall", f"{basic['Recall']:.4f}", f"{get_delta('Recall'):.4f}")
        c4.metric("F1-Score", f"{basic['F1-Score']:.4f}", f"{get_delta('F1-Score'):.4f}")
        c5.metric("ROC-AUC", f"{basic['ROC-AUC']:.4f}", f"{get_delta('ROC-AUC'):.4f}")

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

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{lr_smote_row['Accuracy']:.4f}", f"{get_delta_lr('Accuracy'):.4f}")
        c2.metric("Precision", f"{lr_smote_row['Precision']:.4f}", f"{get_delta_lr('Precision'):.4f}")
        c3.metric("Recall", f"{lr_smote_row['Recall']:.4f}", f"{get_delta_lr('Recall'):.4f}")
        c4.metric("F1-Score", f"{lr_smote_row['F1-Score']:.4f}", f"{get_delta_lr('F1-Score'):.4f}")
        c5.metric("ROC-AUC", f"{lr_smote_row['ROC-AUC']:.4f}", f"{get_delta_lr('ROC-AUC'):.4f}")

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

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Accuracy", f"{rf_smote['Accuracy']:.4f}", f"{get_rf_delta('Accuracy'):.4f}")
        r2.metric("Precision", f"{rf_smote['Precision']:.4f}", f"{get_rf_delta('Precision'):.4f}")
        r3.metric("Recall", f"{rf_smote['Recall']:.4f}", f"{get_rf_delta('Recall'):.4f}")
        r4.metric("F1-Score", f"{rf_smote['F1-Score']:.4f}", f"{get_rf_delta('F1-Score'):.4f}")
        r5.metric("ROC-AUC", f"{rf_smote['ROC-AUC']:.4f}", f"{get_rf_delta('ROC-AUC'):.4f}")

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

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Accuracy", f"{dt_smote['Accuracy']:.4f}", f"{get_dt_delta('Accuracy'):.4f}")
        d2.metric("Precision", f"{dt_smote['Precision']:.4f}", f"{get_dt_delta('Precision'):.4f}")
        d3.metric("Recall", f"{dt_smote['Recall']:.4f}", f"{get_dt_delta('Recall'):.4f}")
        d4.metric("F1-Score", f"{dt_smote['F1-Score']:.4f}", f"{get_dt_delta('F1-Score'):.4f}")
        d5.metric("ROC-AUC", f"{dt_smote['ROC-AUC']:.4f}", f"{get_dt_delta('ROC-AUC'):.4f}")

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


elif page == "🔬 Robustness Checks":
    st.title("🔬 Robustness Checks")

    robustness_df = run_robustness_checks(X, y)

    st.subheader("📈 ROC-AUC Across All Checks")
    st.pyplot(plot_robustness_roc_auc(robustness_df))

    st.divider()

    st.subheader("📋 Full Results")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    st.dataframe(
        robustness_df.style.format({c: "{:.4f}" for c in metric_cols}),
        width="stretch",
        hide_index=True,
    )