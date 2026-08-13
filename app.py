
import streamlit as st
import pandas as pd

import data_preprocessing as dp
import data_visualization as dv
import knn as km

st.set_page_config(page_title="Heart Disease Risk -- KNN", layout="wide", page_icon="\u2764\ufe0f")

DEFAULT_DATA_PATH = "heart_disease.csv"


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading & preprocessing data...")
def load_pipeline_data(path):
    return dp.run_pipeline(path)


@st.cache_data(show_spinner="Loading raw data...")
def load_raw_data(path):
    df = dp.load_data(path)
    numeric_cols, categorical_cols = dp.get_column_groups(df)
    return df, numeric_cols, categorical_cols


@st.cache_resource(show_spinner="Training Basic KNN & SMOTE KNN (grid search, one-time)...")
def train_models(X, y):
    X_train, X_test, y_train, y_test = km.get_70_30_split(X, y)
    basic = km.tune_and_evaluate(
        km.build_basic_pipeline(), km.BASIC_PARAM_GRID,
        X_train, X_test, y_train, y_test, "1. Basic KNN",
    )
    smote = km.tune_and_evaluate(
        km.build_smote_pipeline(), km.SMOTE_PARAM_GRID,
        X_train, X_test, y_train, y_test, "2. SMOTE KNN",
    )
    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "results": {"1. Basic KNN": basic, "2. SMOTE KNN": smote},
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("\u2764\ufe0f Heart Disease KNN")
uploaded = st.sidebar.file_uploader("Upload heart_disease.csv (optional)", type="csv")
data_source = uploaded if uploaded is not None else DEFAULT_DATA_PATH

page = st.sidebar.radio(
    "Navigate",
    ["\U0001F3E0 Overview", "\U0001F50D EDA", "\U0001F9F9 Preprocessing", "\U0001F4CA Model Comparison", "\U0001F52E Predict"],
)

# ---------------------------------------------------------------------------
# Load data + train models (cached -- cheap after first run)
# ---------------------------------------------------------------------------

try:
    raw_df, numeric_cols, categorical_cols = load_raw_data(data_source)
    pipeline_data = load_pipeline_data(data_source)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DEFAULT_DATA_PATH}` next to app.py. "
        "Upload a CSV from the sidebar to continue."
    )
    st.stop()

X, y = pipeline_data["X"], pipeline_data["y"]
le_target = pipeline_data["le_target"]
missing_treatment_summary = pipeline_data["missing_treatment_summary"]
target_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))

trained = train_models(X, y)
results = trained["results"]
y_test = trained["y_test"]
results_df = pd.DataFrame([res["metrics"] for res in results.values()])


# =============================================================================
# PAGE: Overview
# =============================================================================
if page == "\U0001F3E0 Overview":
    st.title("Heart Disease Risk Prediction")
    st.caption("Basic KNN vs SMOTE-balanced KNN \u2014 both tuned & evaluated on a 70/30 split")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", raw_df.shape[0])
    c2.metric("Raw columns", raw_df.shape[1])
    c3.metric("Numeric features", len(numeric_cols))
    c4.metric("Categorical features", len(categorical_cols))

    st.subheader("Sample of raw data")
    st.dataframe(raw_df.head(10), width='stretch')

    st.subheader("Target class balance")
    counts = raw_df[dp.TARGET_COL].value_counts()
    c1, c2 = st.columns([1, 2])
    c1.dataframe(counts)
    c2.bar_chart(counts)

    st.subheader("Best model so far")
    best_name = results_df.loc[results_df["ROC-AUC"].idxmax(), "Model"]
    best_auc = results_df["ROC-AUC"].max()
    st.success(f"**{best_name}** currently has the higher test ROC-AUC ({best_auc:.3f}). See *Model Comparison* for full detail.")


# =============================================================================
# PAGE: EDA
# =============================================================================
elif page == "\U0001F50D EDA":
    st.title("Exploratory Data Analysis")
    st.caption("All plots use the raw (uncleaned) dataset.")

    st.subheader("Class Distribution")
    st.pyplot(dv.plot_class_distribution(raw_df))

    st.subheader("Numeric Feature Distributions")
    st.pyplot(dv.plot_numeric_distributions(raw_df, numeric_cols))

    st.subheader("QQ-Plots (vs. Uniform)")
    st.pyplot(dv.plot_qq_plots(raw_df, numeric_cols))

    st.subheader("Categorical Feature Distributions (%)")
    st.pyplot(dv.plot_categorical_distributions(raw_df, categorical_cols))

    st.subheader("Correlation Heatmap (numeric features)")
    st.pyplot(dv.plot_correlation_heatmap(raw_df, numeric_cols))

    st.subheader("Outliers (1.5\u00d7 IQR)")
    st.pyplot(dv.plot_outliers_boxplot(raw_df, numeric_cols))
    st.dataframe(dv.compute_outlier_counts(raw_df, numeric_cols), width='stretch')

    st.subheader("Feature vs Target Associations")
    table_numeric, table_categorical, fig_assoc = dv.test_target_associations(raw_df, numeric_cols, categorical_cols)
    st.pyplot(fig_assoc)
    colA, colB = st.columns(2)
    colA.markdown("**Numeric (point-biserial r)**")
    colA.dataframe(table_numeric, width='stretch')
    colB.markdown("**Categorical (Cramer's V)**")
    colB.dataframe(table_categorical, width='stretch')

    st.subheader("Numeric Features by Target")
    st.pyplot(dv.plot_numeric_by_target(raw_df, numeric_cols))

    st.subheader("Heart Disease Rate by Category")
    st.pyplot(dv.plot_categorical_rate_by_target(raw_df, categorical_cols))

    with st.expander("More: raw counts & normalized % by category"):
        st.pyplot(dv.plot_categorical_counts_by_target(raw_df, categorical_cols))
        st.pyplot(dv.plot_categorical_percentage_by_target(raw_df, categorical_cols))


# =============================================================================
# PAGE: Preprocessing
# =============================================================================
elif page == "\U0001F9F9 Preprocessing":
    st.title("Preprocessing")

    st.subheader("Missing Values (raw data)")
    df_missing, fig_missing = dv.plot_missing_values(raw_df)
    if not df_missing.empty:
        st.pyplot(fig_missing)
        st.dataframe(df_missing, width='stretch')
    else:
        st.info("No missing values found in the raw data.")

    st.subheader("Is 'Alcohol Consumption' missing at random (MCAR)?")
    mcar_df, fig_mcar = dv.test_alcohol_missingness_mcar(raw_df, numeric_cols, categorical_cols)
    st.pyplot(fig_mcar)
    st.dataframe(mcar_df, width='stretch')

    st.subheader("Missing-Value Treatment Applied")
    display_summary = missing_treatment_summary.copy()
    display_summary["Imputation Value"] = display_summary["Imputation Value"].astype(str)
    st.dataframe(display_summary, width='stretch')
    st.caption(
        "Alcohol Consumption is kept as its own 'Unknown' category (MCAR, too much missing "
        "to safely impute). Everything else uses median (numeric) / mode (categorical) imputation."
    )

    st.subheader("Encoding")
    st.markdown(
        "- **Ordinal** (`Exercise Habits`, `Stress Level`, `Sugar Consumption`, `Alcohol Consumption`) "
        "\u2192 integer-mapped, preserving order.\n"
        "- **Binary / nominal** (Gender, Smoking, Family Heart Disease, Diabetes, "
        "High Blood Pressure, Low/High Cholesterol) \u2192 one-hot encoded, so KNN's distance metric "
        "doesn't read a false rank order into them.\n"
        f"- **Target** (`{dp.TARGET_COL}`) \u2192 label-encoded: `{target_mapping}`"
    )
    st.write(f"Final feature matrix: **{X.shape[0]} rows \u00d7 {X.shape[1]} columns**")
    st.dataframe(X.head(10), width='stretch')

    st.subheader("Feature-Target Diagnostics (post-encoding)")
    fig_corr, target_corr = dv.plot_target_correlation_heatmap(pipeline_data["df"], X.columns.tolist())
    st.pyplot(fig_corr)

    anova_df = dv.compute_anova_scores(X, y)
    chi2_df = dv.compute_chi2_scores(X, y)
    colA, colB = st.columns(2)
    colA.markdown("**ANOVA F-scores**")
    colA.dataframe(anova_df, width='stretch')
    colB.markdown("**Chi-Square scores**")
    colB.dataframe(chi2_df, width='stretch')


# =============================================================================
# PAGE: Model Comparison
# =============================================================================
elif page == "\U0001F4CA Model Comparison":
    st.title("Model Comparison \u2014 Basic KNN vs SMOTE KNN")
    st.caption("Both tuned with 5-fold GridSearchCV (scoring = roc_auc) on the same 70/30 stratified split.")

    st.dataframe(results_df.drop(columns=["Best Params"]), width='stretch')

    tabs = st.tabs(list(results.keys()))
    for tab, name in zip(tabs, results.keys()):
        with tab:
            res = results[name]
            st.write(f"**Best Params:** `{res['metrics']['Best Params']}`")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", res["metrics"]["Accuracy"])
            c2.metric("Precision", res["metrics"]["Precision"])
            c3.metric("Recall", res["metrics"]["Recall"])
            c4.metric("F1-Score", res["metrics"]["F1-Score"])
            c5.metric("ROC-AUC", res["metrics"]["ROC-AUC"])
            st.text(res["classification_report"])

    st.subheader("Confusion Matrices")
    st.pyplot(km.plot_confusion_matrices(results))

    st.subheader("ROC Curves")
    st.pyplot(km.plot_roc_curves(results, y_test))

    st.subheader("Metric Comparison")
    st.pyplot(km.plot_metric_comparison(results_df))


# =============================================================================
# PAGE: Predict
# =============================================================================
elif page == "\U0001F52E Predict":
    st.title("Predict Heart Disease Risk")
    st.caption("Fill in the fields, pick a model, and get a live prediction.")

    model_choice = st.selectbox("Model", list(results.keys()))
    best_model = results[model_choice]["best_model"]

    with st.form("predict_form"):
        st.subheader("Numeric")
        numeric_inputs = {}
        cols = st.columns(3)
        for i, col in enumerate(numeric_cols):
            default = float(raw_df[col].median())
            numeric_inputs[col] = cols[i % 3].number_input(col, value=default)

        st.subheader("Categorical")
        categorical_inputs = {}
        cols2 = st.columns(3)
        for i, col in enumerate(categorical_cols):
            if col in dp.ORDINAL_MAPS:
                options = list(dp.ORDINAL_MAPS[col].keys())
            else:
                options = sorted(raw_df[col].dropna().unique().tolist())
            categorical_inputs[col] = cols2[i % 3].selectbox(col, options)

        submitted = st.form_submit_button("Predict")

    if submitted:
        raw_input = {**numeric_inputs, **categorical_inputs}
        row = dp.build_single_row_features(raw_input, categorical_cols, X.columns.tolist())

        pred = int(best_model.predict(row)[0])
        prob_disease = float(best_model.predict_proba(row)[0, 1])
        label = le_target.inverse_transform([pred])[0]

        st.subheader("Result")
        if pred == 1:
            st.error(f"Prediction: **{label}** \u2014 probability of heart disease: {prob_disease:.1%}")
        else:
            st.success(f"Prediction: **{label}** \u2014 probability of heart disease: {prob_disease:.1%}")
        st.progress(min(max(prob_disease, 0.0), 1.0))
        st.caption(
            f"Model: {model_choice} \u2014 Best Params: `{results[model_choice]['metrics']['Best Params']}`"
        )