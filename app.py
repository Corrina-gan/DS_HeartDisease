
import streamlit as st
import pandas as pd

import data_preprocessing as dp
import data_visualization as dv
import plotly.graph_objects as go
import knn as km
import logistic_regression as lgm

st.set_page_config(page_title="Heart Disease Risk", layout="wide", page_icon="\u2764\ufe0f")

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


@st.cache_resource(show_spinner="Training Basic & SMOTE Logistic Regression (grid search, one-time)...")
def train_logreg_models(X, y):
    X_train, X_test, y_train, y_test = lgm.get_70_30_split(X, y)
    basic = lgm.tune_and_evaluate(
        lgm.build_basic_pipeline(), lgm.BASIC_PARAM_GRID,
        X_train, X_test, y_train, y_test, "1. Basic Logistic Regression",
    )
    smote = lgm.tune_and_evaluate(
        lgm.build_smote_pipeline(), lgm.SMOTE_PARAM_GRID,
        X_train, X_test, y_train, y_test, "2. SMOTE Logistic Regression",
    )
    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "results": {"1. Basic Logistic Regression": basic, "2. SMOTE Logistic Regression": smote},
    }


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    # Inject Custom CSS to make sidebar menu items large, wide, and very easy to click
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

    # Centered, styled title for a professional clinic feel
    st.markdown("<h2 style='text-align: center; color: #c44e52;'>❤️ Heart Disease Prediction</h2>", unsafe_allow_html=True)    
    st.divider()
    st.markdown("### 🧭 Main Menu")
    
    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Home (Predict & Overview)", 
            "🔍 EDA", 
            "🧹 Preprocessing", 
            "📊 Model Comparison"
        ],
        label_visibility="collapsed"
    )

# ---------------------------------------------------------------------------
# Load data + PREFETCH Heavy Computations
# ---------------------------------------------------------------------------

try:
    # Directly load from the hardcoded local CSV file
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

trained = train_models(X, y)
results = trained["results"]
y_test = trained["y_test"]
results_df = pd.DataFrame([res["metrics"] for res in results.values()])

trained_lr = train_logreg_models(X, y)
results_lr = trained_lr["results"]
y_test_lr = trained_lr["y_test"]
results_df_lr = pd.DataFrame([res["metrics"] for res in results_lr.values()])

# Combined view across all trained model families, used by the Home predictor
# and the dataset overview so they reflect the current best model overall.
all_results = {**results, **results_lr}
all_results_df = pd.concat([results_df, results_df_lr], ignore_index=True)

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
# PAGE: Home (Predict & Overview)
# =============================================================================
if page == "🏠 Home (Predict & Overview)":
    st.title("❤️ Heart Disease Risk Dashboard")
    st.markdown("Use the calculator below to assess patient risk, or scroll down to view the dataset overview.")

    # -------------------------------------------------------------------------
    # SECTION 1: LIVE PREDICTOR
    # -------------------------------------------------------------------------
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
            submitted = st.form_submit_button("Generate Prediction", use_container_width=True)

    if submitted:
        raw_input = {**numeric_inputs, **categorical_inputs}
        row = dp.build_single_row_features(raw_input, categorical_cols, X.columns.tolist())

        pred = int(best_model.predict(row)[0])
        prob_disease = float(best_model.predict_proba(row)[0, 1])
        label = le_target.inverse_transform([pred])[0]

        st.divider()
        st.subheader("Prediction Result")
        
        # Display the text result
        if pred == 1:
            st.error(f"### ⚠️ Prediction: {label}")
        else:
            st.success(f"### ✅ Prediction: {label}")
        
        # Create a compact, beautifully proportioned Plotly Gauge Chart
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
                    {'range': [0, 30], 'color': "#d4edda"},   # Green (Low Risk)
                    {'range': [30, 70], 'color': "#ffeeba"},  # Yellow (Moderate Risk)
                    {'range': [70, 100], 'color': "#f8d7da"}  # Red (High Risk)
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': prob_disease * 100
                }
            }
        ))
        
        # Shrink the overall height and tighten margins so it isn't massive
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        
        # Center the gauge chart nicely
        g_col1, g_col2, g_col3 = st.columns([1, 2, 1])
        with g_col2:
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.write(f"<div style='text-align: center; color: #888;'>Powered by: {model_choice} | Best Params: {all_results[model_choice]['metrics']['Best Params']}</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 2: DATASET OVERVIEW
    # -------------------------------------------------------------------------
    st.divider()
    st.header("📊 Dataset Overview")

    # High-level Dataset Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patient Records", f"{raw_df.shape[0]:,}")
    c2.metric("Total Attributes", raw_df.shape[1])
    c3.metric("Numeric Features", len(numeric_cols))
    c4.metric("Categorical Features", len(categorical_cols))

    st.markdown("<br>", unsafe_allow_html=True)

    overview_col1, overview_col2 = st.columns([1, 1.5])
    
    with overview_col1:
        st.markdown("**Target Class Balance**")
        st.pyplot(eda_figs["class_dist"]) 

    with overview_col2:
        st.markdown("**Current Top Performing Model**")
        best_name = all_results_df.loc[all_results_df["ROC-AUC"].idxmax(), "Model"]
        best_auc = all_results_df["ROC-AUC"].max()
        st.info(f"🏆 **{best_name}** currently leads with a test ROC-AUC of **{best_auc:.3f}**.")
        
        # Tidy up the UI by hiding the raw data inside an expander
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📂 View Sample Patient Records (Raw Data)"):
            st.dataframe(raw_df.head(15), use_container_width=True)

# =============================================================================
# PAGE: EDA
# =============================================================================
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Explore the underlying patterns in the raw dataset before any cleaning or encoding.")

    # Organize the massive list of charts into 3 logical, clickable tabs
    eda_tab1, eda_tab2, eda_tab3 = st.tabs([
        "📊 Distributions", 
        "🎯 Target Associations", 
        "🔬 Data Quality & Outliers"
    ])

    with eda_tab1:
        st.subheader("Feature & Class Distributions")
        # Put Class Distribution and Categorical Dist side-by-side
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
        colA.dataframe(table_numeric, use_container_width=True)
        colB.markdown("**Categorical (Cramer's V)**")
        colB.dataframe(table_categorical, use_container_width=True)

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
            st.dataframe(outlier_df, use_container_width=True)


# =============================================================================
# PAGE: Preprocessing
# =============================================================================
elif page == "🧹 Preprocessing":
    st.title("🧹 Data Preprocessing")

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
            st.dataframe(display_summary, use_container_width=True)
            
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
                "- **Features:** `Gender`, `Smoking`, `Diabetes`, `Blood Pressure`, `Cholesterol`, `Family History`\n\n"
                "**3. Label Encoding**\n"
                f"- **Rule:** Converts the final predicted outcome into a binary machine-readable format.\n"
                f"- **Target:** `{dp.TARGET_COL}` → `{target_mapping}`"
            )
        with col_e2:
            st.write(f"**Final Feature Matrix:** {X.shape[0]:,} rows × {X.shape[1]} columns")
            st.dataframe(X.head(10), use_container_width=True)

    with prep_tab3:
        st.subheader("Post-Encoding Feature Diagnostics")
        st.pyplot(fig_corr)

        col_d1, col_d2 = st.columns(2)
        col_d1.markdown("**ANOVA F-scores**")
        col_d1.dataframe(anova_df, use_container_width=True)
        col_d2.markdown("**Chi-Square scores**")
        col_d2.dataframe(chi2_df, use_container_width=True)

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
        # 1. EXECUTIVE SUMMARY (Moved to Top)
        # ==========================================
        st.info(
            "**📝 Executive Summary & Report Conclusion:**\n\n"
            "While the **Basic KNN** achieves a higher overall Accuracy, "
            "it struggles to identify patients with heart disease (extremely low Recall). "
            "Applying **SMOTE** successfully balances the training data, significantly boosting the model's "
            "Recall for positive cases. In medical diagnostics, minimizing False Negatives is critical, "
            "making the **SMOTE-tuned KNN** the superior practical model despite the drop in raw accuracy."
        )

        st.divider()

        # ==========================================
        # 2. HIGH-LEVEL METRICS IMPACT
        # ==========================================
        st.subheader("📊 High-Level Metrics Impact (SMOTE vs. Basic)")
        
        basic = results_df.iloc[0]
        smote = results_df.iloc[1]
        
        def get_delta(metric):
            return float(smote[metric] - basic[metric])

        # Large Metric Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{smote['Accuracy']:.4f}", f"{get_delta('Accuracy'):.4f}")
        c2.metric("Precision", f"{smote['Precision']:.4f}", f"{get_delta('Precision'):.4f}")
        c3.metric("Recall", f"{smote['Recall']:.4f}", f"{get_delta('Recall'):.4f}")
        c4.metric("F1-Score", f"{smote['F1-Score']:.4f}", f"{get_delta('F1-Score'):.4f}")
        c5.metric("ROC-AUC", f"{smote['ROC-AUC']:.4f}", f"{get_delta('ROC-AUC'):.4f}")

        st.divider()

        # ==========================================
        # 3. VISUAL EVALUATION (Organized into Tabs)
        # ==========================================
        st.subheader("📈 Visual Evaluation")
        st.write("Navigate through the tabs below to explore the model visualizations.")
        
        # Neatly organize the heavy graphs into clickable tabs to save vertical space
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

        # ==========================================
        # 4. DETAILED CLASS BREAKDOWN (Side-by-Side)
        # ==========================================
        st.subheader("🔍 Class-by-Class Breakdown")
        
        col_basic, col_smote = st.columns(2)
        models_list = list(results.keys()) 
        
        # --- LEFT COLUMN: BASIC KNN ---
        with col_basic:
            res_basic = results[models_list[0]]
            st.markdown(f"### 🔵 {models_list[0]}")
            st.info(f"**🎯 Overall Accuracy:** {res_basic['metrics']['Accuracy']:.2%} &nbsp; | &nbsp; **📈 ROC-AUC:** {res_basic['metrics']['ROC-AUC']:.2%}")
            
            report_basic = km.classification_report(y_test, res_basic["y_pred"], output_dict=True, zero_division=0)
            df_rep_basic = pd.DataFrame(report_basic).transpose()
            df_rep_basic.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'}, inplace=True)
            df_rep_basic = df_rep_basic.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')
            
            st.dataframe(df_rep_basic.style.background_gradient(cmap='Blues').format("{:.3f}"), use_container_width=True)
            
            # Hide hyperparameters in a neat dropdown
            with st.expander("⚙️ View Basic Hyperparameters"):
                st.json(res_basic['metrics']['Best Params'])

        # --- RIGHT COLUMN: SMOTE KNN ---
        with col_smote:
            res_smote = results[models_list[1]]
            st.markdown(f"### 🟢 {models_list[1]}")
            st.success(f"**🎯 Overall Accuracy:** {res_smote['metrics']['Accuracy']:.2%} &nbsp; | &nbsp; **📈 ROC-AUC:** {res_smote['metrics']['ROC-AUC']:.2%}")
            
            report_smote = km.classification_report(y_test, res_smote["y_pred"], output_dict=True, zero_division=0)
            df_rep_smote = pd.DataFrame(report_smote).transpose()
            df_rep_smote.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'}, inplace=True)
            df_rep_smote = df_rep_smote.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')
            
            st.dataframe(df_rep_smote.style.background_gradient(cmap='Greens').format("{:.3f}"), use_container_width=True)

            # Hide hyperparameters in a neat dropdown
            with st.expander("⚙️ View SMOTE Hyperparameters"):
                st.json(res_smote['metrics']['Best Params'])

    # -------------------------------------------------------------------------
    # TAB 2: Logistic Regression
    # -------------------------------------------------------------------------
    with model_tabs[1]:
        st.header("Logistic Regression")

        # ==========================================
        # 1. EXECUTIVE SUMMARY
        # ==========================================
        lr_basic_row = results_df_lr.iloc[0]
        lr_smote_row = results_df_lr.iloc[1]
        recall_gain = float(lr_smote_row["Recall"] - lr_basic_row["Recall"])
        st.info(
            "**📝 Executive Summary & Report Conclusion:**\n\n"
            "Logistic Regression provides a fully interpretable baseline: each feature gets a signed "
            "coefficient showing exactly how it moves risk up or down after standardization. "
            f"Applying **SMOTE** shifts Recall by **{recall_gain:+.2%}**, trading some precision for "
            "better detection of at-risk patients. Compared to KNN, Logistic Regression is the model of "
            "choice when clinicians need to explain *why* a prediction was made, not just what it is."
        )

        st.divider()

        # ==========================================
        # 2. HIGH-LEVEL METRICS IMPACT
        # ==========================================
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

        # ==========================================
        # 3. VISUAL EVALUATION
        # ==========================================
        st.subheader("📈 Visual Evaluation")
        st.write("Navigate through the tabs below to explore the model visualizations.")

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

        # ==========================================
        # 4. DETAILED CLASS BREAKDOWN
        # ==========================================
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

            st.dataframe(df_rep_lr_basic.style.background_gradient(cmap='Blues').format("{:.3f}"), use_container_width=True)

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

            st.dataframe(df_rep_lr_smote.style.background_gradient(cmap='Greens').format("{:.3f}"), use_container_width=True)

            with st.expander("⚙️ View SMOTE Hyperparameters"):
                st.json(res_lr_smote['metrics']['Best Params'])

        st.divider()

        # ==========================================
        # 5. COEFFICIENT INTERPRETABILITY (LR-specific)
        # ==========================================
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
                    use_container_width=True,
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
                    use_container_width=True,
                    hide_index=True,
                )

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