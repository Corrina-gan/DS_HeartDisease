"""
Exploratory data analysis and visualizations for the Heart Disease dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import MinMaxScaler

from data_preprocessing import load_data, get_column_groups, TARGET_COL, ALCOHOL_COL

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7, 5)

def section(title):
    width = 70
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)

def plot_class_distribution(df, target_col=TARGET_COL):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(x=target_col, hue=target_col, data=df, palette="Set2", legend=False, ax=ax)
    ax.set_title("Class distribution: Heart Disease Status")
    fig.tight_layout()
    return fig

def plot_numeric_distributions(df, numeric_cols):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], bins=30, kde=True, color="steelblue", edgecolor="black", ax=axes[i])
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].grid(axis="y", alpha=0.3)
    fig.suptitle("Distribution of Numerical Variables", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig

def plot_qq_plots(df, numeric_cols):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        data = df[col].dropna()
        a, b = data.min(), data.max()
        stats.probplot(data, dist=stats.uniform(loc=a, scale=b - a), plot=axes[i])
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].get_lines()[0].set_markerfacecolor("steelblue")
        axes[i].get_lines()[0].set_markersize(3)
    fig.suptitle("QQ-Plots: Observed Quantiles vs. Theoretical Uniform Distribution", fontsize=14)
    fig.tight_layout()
    return fig

def plot_categorical_distributions(df, categorical_cols):
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    axes = axes.flatten()
    for i, col in enumerate(categorical_cols):
        df_pct = df[col].value_counts(normalize=True).mul(100).reset_index()
        df_pct.columns = [col, "Percentage"]
        sns.barplot(data=df_pct, x="Percentage", hue=col, legend=False, y=col, ax=axes[i], palette="Blues_r")
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Percentage (%)", fontsize=9)
        axes[i].set_ylabel("")
        axes[i].set_xlim(0, 70)
        for p in axes[i].patches:
            width = p.get_width()
            axes[i].annotate(f"{width:.1f}%", (width + 1.5, p.get_y() + p.get_height() / 2.0), ha="left", va="center", fontsize=9)
    for j in range(len(categorical_cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Categorical Variable Distributions (%)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig

def plot_correlation_heatmap(df, numeric_cols):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Attributes")
    fig.tight_layout()
    return fig

def check_data_quality(df, categorical_cols):
    """Structural quality checks (report Section 3.5.1 / Figure 3.10):
    duplicate records and the unique values each categorical/binary
    attribute actually takes, to catch inconsistent labels (e.g. "Yes" vs
    "yes")."""
    n_duplicates = int(df.duplicated().sum())
    unique_values = pd.DataFrame([
        {
            "Attribute": col,
            "Unique Values": ", ".join(sorted(str(v) for v in df[col].dropna().unique())),
            "Category Count": df[col].nunique(dropna=True),
        }
        for col in categorical_cols
    ])
    return n_duplicates, unique_values


def plot_missing_values(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
    df_missing = missing_summary[missing_summary["missing_count"] > 0].sort_values("missing_%", ascending=False).reset_index()
    df_missing.rename(columns={"index": "Variable"}, inplace=True)
    if df_missing.empty:
        return df_missing, None
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#c44e52" if i == 0 else "#4c72b0" for i in range(len(df_missing))]
    bars = ax.barh(df_missing["Variable"], df_missing["missing_%"], color=colors, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Missing (%)")
    ax.set_title("Percentage of Missing Values by Variable", fontweight="bold", pad=20)
    max_missing = df_missing["missing_%"].max()
    ax.set_xlim(0, max_missing + 5)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (max_missing * 0.02), bar.get_y() + bar.get_height() / 2, f"{width:.2f}%", va="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return df_missing, fig

def test_alcohol_missingness_mcar(df, numeric_cols, categorical_cols, target_col=TARGET_COL, alcohol_col=ALCOHOL_COL):
    is_missing = df[alcohol_col].isnull()
    mcar_rows = []
    for col in numeric_cols:
        valid = df[col].notnull()
        stat, p = stats.ttest_ind(df.loc[valid & is_missing, col], df.loc[valid & ~is_missing, col], equal_var=False)
        mcar_rows.append({"variable": col, "test": "t-test", "statistic": round(stat, 3), "p_value": round(p, 4)})
    for col in [c for c in categorical_cols if c != alcohol_col] + [target_col]:
        valid = df[col].notnull()
        ct = pd.crosstab(is_missing[valid], df.loc[valid, col])
        chi2_val, p, dof, exp = stats.chi2_contingency(ct)
        mcar_rows.append({"variable": col, "test": "chi-square", "statistic": round(chi2_val, 3), "p_value": round(p, 4)})
    mcar_df = pd.DataFrame(mcar_rows).sort_values("p_value").reset_index(drop=True)
    fig = _plot_mcar_lollipop(mcar_df, target_col)
    return mcar_df, fig

def _plot_mcar_lollipop(mcar_df, target_col):
    mcar_plot_df = mcar_df.copy()
    mcar_plot_df["neg_log10_p"] = -np.log10(mcar_plot_df["p_value"])
    mcar_plot_df["is_target"] = mcar_plot_df["variable"] == target_col
    mcar_plot_df = mcar_plot_df.sort_values("p_value", ascending=False).reset_index(drop=True)
    bonf_threshold_mcar = 0.05 / len(mcar_plot_df)
    colors = mcar_plot_df["is_target"].map({True: "crimson", False: "steelblue"})
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hlines(y=mcar_plot_df["variable"], xmin=0, xmax=mcar_plot_df["neg_log10_p"], color="lightgrey", linewidth=1)
    ax.scatter(mcar_plot_df["neg_log10_p"], mcar_plot_df["variable"], color=colors, s=70, zorder=3)
    ax.axvline(-np.log10(0.05), color="orange", linestyle="--", linewidth=1, label="Raw threshold (p = 0.05)")
    ax.axvline(-np.log10(bonf_threshold_mcar), color="crimson", linestyle="--", linewidth=1, label="Bonferroni threshold")
    ax.set_xlabel("-log10(p-value)\n(further right = stronger evidence against MCAR)")
    ax.set_ylabel("Variable")
    ax.set_title("MCAR Test: Association Between Alcohol Consumption Missingness\nand Observed Variables")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig

def plot_outliers_boxplot(df, numeric_cols):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=df[numeric_cols], orient="h", ax=ax)
    ax.set_title("Boxplot of Numerical Features")
    ax.set_xlabel("Value")
    ax.set_ylabel("Features")
    fig.tight_layout()
    return fig

def compute_outlier_counts(df, numeric_cols):
    outlier_counts = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[col] = ((df[col] < lb) | (df[col] > ub)).sum()
    return pd.Series(outlier_counts, name="n_outliers").to_frame()

def test_target_associations(df, numeric_cols, categorical_cols, target_col=TARGET_COL):
    y_temp = df[target_col].map({"No": 0, "Yes": 1})
    pb_rows = []
    for col in numeric_cols:
        valid = df[col].notnull() & y_temp.notnull()
        r, p = stats.pointbiserialr(y_temp[valid], df.loc[valid, col])
        pb_rows.append({"variable": col, "point_biserial_r": round(r, 4), "p_value": round(p, 4)})
    table_numeric = pd.DataFrame(pb_rows).sort_values("p_value").reset_index(drop=True)

    chi_rows = []
    for col in categorical_cols:
        ct = pd.crosstab(df[col], df[target_col])
        chi2_val, p, dof, exp = stats.chi2_contingency(ct)
        n_obs = ct.sum().sum()
        cramers_v = np.sqrt(chi2_val / (n_obs * (min(ct.shape) - 1)))
        chi_rows.append({"variable": col, "chi2": round(chi2_val, 3), "p_value": round(p, 4), "cramers_v": round(cramers_v, 4)})
    table_categorical = pd.DataFrame(chi_rows).sort_values("p_value").reset_index(drop=True)

    fig = _plot_effect_sizes(table_numeric, table_categorical)
    return table_numeric, table_categorical, fig

def _plot_effect_sizes(table_numeric, table_categorical):
    numeric_effect = table_numeric[["variable", "point_biserial_r", "p_value"]].copy()
    numeric_effect["effect_size"] = numeric_effect["point_biserial_r"].abs()
    numeric_effect = numeric_effect[["variable", "effect_size", "p_value"]]

    categorical_effect = table_categorical[["variable", "cramers_v", "p_value"]].copy()
    categorical_effect = categorical_effect.rename(columns={"cramers_v": "effect_size"})

    effect_df = pd.concat([numeric_effect, categorical_effect], ignore_index=True)
    bonf_threshold = 0.05 / len(effect_df)
    effect_df["significance"] = "Not significant"
    effect_df.loc[effect_df["p_value"] < 0.05, "significance"] = "Raw p < 0.05"
    effect_df.loc[effect_df["p_value"] < bonf_threshold, "significance"] = "Bonferroni significant"
    effect_df = effect_df.sort_values("effect_size", ascending=False)

    palette = {"Not significant": "lightgrey", "Raw p < 0.05": "orange", "Bonferroni significant": "crimson"}
    fig, ax = plt.subplots(figsize=(7, 8))
    sns.barplot(data=effect_df, x="effect_size", y="variable", hue="significance", palette=palette, dodge=False, ax=ax)
    ax.axvline(x=0.1, color="black", linestyle="--", label="Small effect (0.1)")
    ax.set_xlabel("Effect Size")
    ax.set_ylabel("")
    ax.set_title("Association Strength Between Predictors and Target")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_numeric_by_target(df, numeric_cols, target_col=TARGET_COL):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(data=df, x=target_col, y=col, hue=target_col, palette=["#4c72b0", "#c44e52"], legend=False, ax=axes[i])
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
    fig.suptitle("Numerical Attributes by Heart Disease Status", fontsize=14)
    fig.tight_layout()
    return fig

def plot_categorical_rate_by_target(df, categorical_cols, target_col=TARGET_COL):
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    axes = axes.flatten()
    for i, col in enumerate(categorical_cols):
        prop = df.groupby(col)[target_col].apply(lambda x: (x == "Yes").mean() * 100).reset_index(name="pct_yes")
        sns.barplot(data=prop, x=col, y="pct_yes", hue=col, legend=False, palette="Blues_d", ax=axes[i])
        axes[i].axhline(20, color="grey", linestyle="--", linewidth=1)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_ylabel("% Heart Disease = Yes")
        axes[i].set_xlabel("")
    for j in range(len(categorical_cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Heart Disease Rate by Category (dashed line = overall 20% base rate)", fontsize=14)
    fig.tight_layout()
    return fig

def plot_categorical_counts_by_target(df, categorical_cols, target_col=TARGET_COL):
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    for ax, col in zip(axes.flatten(), categorical_cols):
        sns.countplot(data=df, x=col, hue=target_col, ax=ax, palette="Set2")
        ax.set_title(f"{col} vs {target_col}")
        ax.tick_params(axis="x", rotation=30)
    axes.flatten()[-1].axis("off")
    fig.tight_layout()
    return fig

def plot_categorical_percentage_by_target(df, categorical_cols, target_col=TARGET_COL):
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    for ax, col in zip(axes.flatten(), categorical_cols):
        temp = df.groupby([col, target_col]).size().reset_index(name="Count")
        temp["Percentage"] = temp.groupby(col)["Count"].transform(lambda x: x / x.sum() * 100)
        sns.barplot(data=temp, x=col, y="Percentage", hue=target_col, ax=ax, palette="Set2")
        ax.axhline(y=20, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{col}")
        ax.set_ylabel("% Heart Disease = Yes")
        ax.set_ylim(0, 25)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.tick_params(axis="x", rotation=30)
        ax.get_legend().remove()
    axes.flatten()[-1].axis("off")
    yes_patch = mpatches.Patch(color=sns.color_palette("Set2")[1], label="Heart Disease: Yes")
    no_patch = mpatches.Patch(color=sns.color_palette("Set2")[0], label="Heart Disease: No")
    base_patch = mpatches.Patch(color="red", label="Overall 20% base rate")
    fig.legend(handles=[yes_patch, no_patch, base_patch], loc="upper right", bbox_to_anchor=(0.98, 1.04), title="Legend")
    fig.suptitle("Heart Disease Rate by Category (dashed line = overall 20% base rate)", y=1.02, fontsize=14)
    fig.tight_layout()
    return fig

def plot_target_correlation_heatmap(df, feature_cols, target_col="target"):
    features = [c for c in feature_cols if c != target_col]
    target_corr = df[features + [target_col]].corr()[target_col].drop(target_col)
    target_corr_sorted = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(target_corr_sorted.to_frame(), annot=True, fmt=".3f", cmap="coolwarm", center=0, cbar=True, ax=ax)
    ax.set_title("Feature Correlation with Target")
    fig.tight_layout()
    return fig, target_corr_sorted

def compute_anova_scores(X, y):
    f_scores, p_values = f_classif(X, y)
    anova_results = pd.DataFrame({
        "Feature": X.columns,
        "F-Score": f_scores,
        "p-value": p_values
    }).sort_values("p-value").reset_index(drop=True)
    return anova_results

def compute_chi2_scores(X, y):
    X_scaled = MinMaxScaler().fit_transform(X)
    chi2_scores, chi2_pvalues = chi2(X_scaled, y)
    chi2_results = pd.DataFrame({
        "Feature": X.columns,
        "Chi2-Score": chi2_scores,
        "p-value": chi2_pvalues
    }).sort_values("p-value").reset_index(drop=True)
    return chi2_results