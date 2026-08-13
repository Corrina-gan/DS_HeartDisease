
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
# Load data + PREFETCH Heavy Computations
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

# --- PREFETCH MAGIC HAPPENS HERE ---
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

# Trigger the prefetch immediately!
eda_figs = prefetch_eda_plots(raw_df, numeric_cols, categorical_cols)
outlier_df, table_numeric, table_categorical, fig_assoc, mcar_df, fig_mcar, fig_corr, anova_df, chi2_df = prefetch_stats(raw_df, numeric_cols, categorical_cols, X, y)

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


# =============================================================================
# PAGE: EDA
# =============================================================================
elif page == "🔍 EDA":
    st.title("Exploratory Data Analysis")
    st.caption("All plots use the raw (uncleaned) dataset.")

    st.subheader("Class Distribution")
    st.pyplot(eda_figs["class_dist"])

    st.subheader("Numeric Feature Distributions")
    st.pyplot(eda_figs["num_dist"])

    st.subheader("QQ-Plots (vs. Uniform)")
    st.pyplot(eda_figs["qq_plots"])

    st.subheader("Categorical Feature Distributions (%)")
    st.pyplot(eda_figs["cat_dist"])

    st.subheader("Correlation Heatmap (numeric features)")
    st.pyplot(eda_figs["corr_heat"])

    st.subheader("Outliers (1.5× IQR)")
    st.pyplot(eda_figs["outliers"])
    st.dataframe(outlier_df, width='stretch')

    st.subheader("Feature vs Target Associations")
    st.pyplot(fig_assoc)
    colA, colB = st.columns(2)
    colA.markdown("**Numeric (point-biserial r)**")
    colA.dataframe(table_numeric, width='stretch')
    colB.markdown("**Categorical (Cramer's V)**")
    colB.dataframe(table_categorical, width='stretch')

    st.subheader("Numeric Features by Target")
    st.pyplot(eda_figs["num_by_target"])

    st.subheader("Heart Disease Rate by Category")
    st.pyplot(eda_figs["cat_rate"])

    with st.expander("More: raw counts & normalized % by category"):
        st.pyplot(eda_figs["cat_counts"])
        st.pyplot(eda_figs["cat_pct"])

# =============================================================================
# PAGE: Preprocessing
# =============================================================================
elif page == "🧹 Preprocessing":
    st.title("Preprocessing")

    st.subheader("Missing Values (raw data)")
    df_missing, fig_missing = dv.plot_missing_values(raw_df)
    if not df_missing.empty:
        st.pyplot(fig_missing)
        st.dataframe(df_missing, width='stretch')
    else:
        st.info("No missing values found in the raw data.")

    st.subheader("Is 'Alcohol Consumption' missing at random (MCAR)?")
    st.pyplot(fig_mcar)
    st.dataframe(mcar_df, width='stretch')

    st.subheader("Missing-Value Treatment Applied")
    display_summary = missing_treatment_summary.copy()
    display_summary["Imputation Value"] = display_summary["Imputation Value"].astype(str)
    st.dataframe(display_summary, width='stretch')
    st.caption("Alcohol Consumption is kept as its own 'Unknown' category (MCAR).")

    st.subheader("Encoding")
    st.markdown("- **Ordinal:** integer-mapped.\n- **Binary / nominal:** one-hot encoded.\n" f"- **Target** (`{dp.TARGET_COL}`) → label-encoded: `{target_mapping}`")
    st.write(f"Final feature matrix: **{X.shape[0]} rows × {X.shape[1]} columns**")
    st.dataframe(X.head(10), width='stretch')

    st.subheader("Feature-Target Diagnostics (post-encoding)")
    st.pyplot(fig_corr)

    colA, colB = st.columns(2)
    colA.markdown("**ANOVA F-scores**")
    colA.dataframe(anova_df, width='stretch')
    colB.markdown("**Chi-Square scores**")
    colB.dataframe(chi2_df, width='stretch')

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
# =============================================================================
# PAGE: Model Comparison
# =============================================================================
elif page == "📊 Model Comparison":
    st.title("Model Comparison")
    st.caption("Evaluate and compare the performance of different models on the 70/30 split.")

    model_tabs = st.tabs(["K-Nearest Neighbors (KNN)", "Logistic Regression", "Random Forest", "Decision Tree"])

    # -------------------------------------------------------------------------
    # TAB 1: KNN (Active Focus)
    # -------------------------------------------------------------------------
    with model_tabs[0]:
        st.header("K-Nearest Neighbors (KNN)")

        # ==========================================
        # 1. THE METRICS SHOWDOWN
        # ==========================================
        st.subheader("📊 Performance Metrics Comparison")
        
        # Color-coded table highlighting the best scores
        metrics_df = results_df.drop(columns=["Best Params", "CV ROC-AUC (best)"]).set_index("Model")
        
        # Define the columns that actually need math formatting
        score_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        
        styled_df = metrics_df.style.highlight_max(
            subset=score_cols, 
            color="#d4edda", 
            axis=0
        ).format("{:.4f}", subset=score_cols) # Added subset here!
        
        st.dataframe(styled_df, use_container_width=True)

        st.markdown("### 🏆 SMOTE vs. Basic Impact")
        
        # Calculate the delta between SMOTE and Basic
        basic = results_df.iloc[0]
        smote = results_df.iloc[1]
        
        def get_delta(metric):
            return float(smote[metric] - basic[metric])

        # Large Metric Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(label="Accuracy (SMOTE)", value=f"{smote['Accuracy']:.4f}", delta=f"{get_delta('Accuracy'):.4f}")
        c2.metric(label="Precision (SMOTE)", value=f"{smote['Precision']:.4f}", delta=f"{get_delta('Precision'):.4f}")
        c3.metric(label="Recall (SMOTE)", value=f"{smote['Recall']:.4f}", delta=f"{get_delta('Recall'):.4f}")
        c4.metric(label="F1-Score (SMOTE)", value=f"{smote['F1-Score']:.4f}", delta=f"{get_delta('F1-Score'):.4f}")
        c5.metric(label="ROC-AUC (SMOTE)", value=f"{smote['ROC-AUC']:.4f}", delta=f"{get_delta('ROC-AUC'):.4f}")

        # ==========================================
        # 2. VISUALIZATIONS
        # ==========================================
        st.divider()
        st.subheader("📈 Visual Evaluation")
        
        # Full-width bar chart
        st.markdown("**Metric Comparison Bar Chart**")
        st.pyplot(km.plot_metric_comparison(results_df))

        st.divider()

        # Let the Confusion Matrices take up the full width so they are big and readable
        st.markdown("**Confusion Matrices**")
        st.pyplot(km.plot_confusion_matrices(results))
        
        st.divider()

        # Put the ROC Curves right below it (also full width)
        st.markdown("**ROC Curves**")
        # To keep the ROC curve from becoming too massive, we can place it in a centered container
        roc_col1, roc_col2, roc_col3 = st.columns([1, 2, 1])
        with roc_col2:
            st.pyplot(km.plot_roc_curves(results, y_test))

# ==========================================
        # 3. DETAILED MODEL REPORTS (SIDE-BY-SIDE)
        # ==========================================
        st.divider()
        st.subheader("🔍 Detailed Model Reports (Side-by-Side)")
        st.write("Compare the exact parameters and class-by-class performance directly.")
        
        # Create two columns for a direct half-and-half comparison
        col_basic, col_smote = st.columns(2)
        
        models_list = list(results.keys()) # ["1. Basic KNN", "2. SMOTE KNN"]
        
        # --- LEFT COLUMN: BASIC KNN ---
        with col_basic:
            res_basic = results[models_list[0]]
            st.markdown(f"### 🔵 {models_list[0]}")
            
            st.markdown("**Best Hyperparameters:**")
            params_basic_df = pd.DataFrame([res_basic['metrics']['Best Params']]).T
            params_basic_df.columns = ["Value"]
            st.dataframe(params_basic_df, use_container_width=True)
            
            # Highlight Overall Metrics cleanly outside the table
            st.info(f"**🎯 Overall Accuracy:** {res_basic['metrics']['Accuracy']:.2%} &nbsp; | &nbsp; **📈 ROC-AUC:** {res_basic['metrics']['ROC-AUC']:.2%}")
            
            st.markdown("**Class-by-Class Breakdown:**")
            report_basic = km.classification_report(y_test, res_basic["y_pred"], output_dict=True, zero_division=0)
            df_rep_basic = pd.DataFrame(report_basic).transpose()
            
            # Rename rows for report clarity and drop the messy 'accuracy' and 'support' rows/columns
            df_rep_basic.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'}, inplace=True)
            df_rep_basic = df_rep_basic.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')
            
            styled_rep_basic = df_rep_basic.style.background_gradient(cmap='Blues').format("{:.3f}")
            st.dataframe(styled_rep_basic, use_container_width=True)

        # --- RIGHT COLUMN: SMOTE KNN ---
        with col_smote:
            res_smote = results[models_list[1]]
            st.markdown(f"### 🟢 {models_list[1]}")
            
            st.markdown("**Best Hyperparameters:**")
            params_smote_df = pd.DataFrame([res_smote['metrics']['Best Params']]).T
            params_smote_df.columns = ["Value"]
            st.dataframe(params_smote_df, use_container_width=True)
            
            # Highlight Overall Metrics cleanly outside the table
            st.success(f"**🎯 Overall Accuracy:** {res_smote['metrics']['Accuracy']:.2%} &nbsp; | &nbsp; **📈 ROC-AUC:** {res_smote['metrics']['ROC-AUC']:.2%}")
            
            st.markdown("**Class-by-Class Breakdown:**")
            report_smote = km.classification_report(y_test, res_smote["y_pred"], output_dict=True, zero_division=0)
            df_rep_smote = pd.DataFrame(report_smote).transpose()
            
            # Rename rows for report clarity and drop the messy 'accuracy' and 'support' rows/columns
            df_rep_smote.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'}, inplace=True)
            df_rep_smote = df_rep_smote.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')
            
            styled_rep_smote = df_rep_smote.style.background_gradient(cmap='Greens').format("{:.3f}")
            st.dataframe(styled_rep_smote, use_container_width=True)

        # ==========================================
        # 4. REPORT CONCLUSION SUMMARY
        # ==========================================
        st.divider()
        st.subheader("📝 Report Conclusion")
        st.info(
            "**Key Finding:** While the Basic KNN achieves higher overall Accuracy, "
            "it fails to reliably identify patients who actually have heart disease (low Recall). "
            "Applying SMOTE successfully balances the training data, significantly boosting the model's "
            "ability to detect true positive cases (Recall). In a medical diagnostic context, minimizing "
            "False Negatives via higher Recall is critical, making the SMOTE-tuned KNN the superior practical model."
        )

    # -------------------------------------------------------------------------
    # TAB 2: Logistic Regression (Placeholder)
    # -------------------------------------------------------------------------
    with model_tabs[1]:
        st.header("Logistic Regression")
        st.info("Space reserved for Basic vs. SMOTE Logistic Regression. \n\n**(Major Strength: Interpretability and Baseline Benchmarking)**")

    # -------------------------------------------------------------------------
    # TAB 3: Random Forest (Placeholder)
    # -------------------------------------------------------------------------
    with model_tabs[2]:
        st.header("Random Forest")
        st.info("Space reserved for Basic vs. SMOTE Random Forest. \n\n**(Major Strength: Pure Predictive Power and Feature Importance)**")

    # -------------------------------------------------------------------------
    # TAB 4: Decision Tree (Placeholder)
    # -------------------------------------------------------------------------
    with model_tabs[3]:
        st.header("Decision Tree")
        st.info("Space reserved for Basic vs. SMOTE Decision Tree. \n\n**(Major Strength: Human-Readable Logic and Non-Linearity)**")