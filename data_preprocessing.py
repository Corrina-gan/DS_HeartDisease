import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TARGET_COL = "Heart Disease Status"
ALCOHOL_COL = "Alcohol Consumption"

# Matches the ordinal mapping from the .ipynb exactly
ORDINAL_MAPS = {
    "Exercise Habits": {"Low": 0, "Medium": 1, "High": 2},
    "Stress Level": {"Low": 0, "Medium": 1, "High": 2},
    "Sugar Consumption": {"Low": 0, "Medium": 1, "High": 2},
    "Alcohol Consumption": {"Unknown": 0, "Low": 1, "Medium": 2, "High": 3},
}


def load_data(path="heart_disease.csv"):
    df = pd.read_csv(path)

    str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip().replace("nan", np.nan)

    df = df.drop_duplicates()
    return df


def get_column_groups(df, target_col=TARGET_COL):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != target_col]
    return numeric_cols, categorical_cols


def handle_missing_values(df, numeric_cols, categorical_cols, alcohol_col=ALCOHOL_COL):
    df = df.copy()
    missing_before = df.isnull().sum()
    remaining_cat_cols = [c for c in categorical_cols if c != alcohol_col]

    df[alcohol_col] = df[alcohol_col].fillna("Unknown")

    num_imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[remaining_cat_cols] = cat_imputer.fit_transform(df[remaining_cat_cols])

    assert df.isnull().sum().sum() == 0

    treatment_map = (
        [(alcohol_col, "Retained as new category", "Unknown")]
        + list(zip(numeric_cols, ["Median imputation"] * len(numeric_cols), num_imputer.statistics_))
        + list(zip(remaining_cat_cols, ["Mode imputation"] * len(remaining_cat_cols), cat_imputer.statistics_))
    )
    missing_treatment_summary = pd.DataFrame([
        {
            "Attribute": col,
            "Missing Count": int(missing_before[col]),
            "Treatment": treatment,
            "Imputation Value": round(value, 4) if isinstance(value, (int, float)) else value,
        }
        for col, treatment, value in treatment_map
    ]).sort_values("Missing Count", ascending=False).reset_index(drop=True)

    return df, missing_treatment_summary


def encode_features(df, categorical_cols, target_col=TARGET_COL, ordinal_maps=None, verbose=True):
    df = df.copy()
    ordinal_maps = ordinal_maps or ORDINAL_MAPS

    # 1. Apply ordinal mapping
    for col, mapping in ordinal_maps.items():
        df[col] = df[col].map(mapping)

    # 2. Get binary columns
    binary_cols = [c for c in categorical_cols if c not in ordinal_maps]
    
    # 3. Create dummy variables (drop_first=True -> true binary encoding,
    # 1 column per Yes/No or Male/Female attribute, avoids the dummy
    # variable trap / redundant mirrored columns)
    df = pd.get_dummies(df, columns=binary_cols, drop_first=True)
    
    # 4. Convert new boolean dummy columns to integers
    new_dummy_cols = [c for c in df.columns if any(c.startswith(b + "_") for b in binary_cols)]
    df[new_dummy_cols] = df[new_dummy_cols].astype(int)
    if verbose:
        print("New columns added:", new_dummy_cols)

    # 5. Encode the target variable
    le_target = LabelEncoder()
    df["target"] = le_target.fit_transform(df[target_col])
    if verbose:
        print("target mapping:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

    # Isolate final feature matrix (X) and target array (y)
    feature_cols = [c for c in df.columns if c not in [target_col, "target"]]
    X = df[feature_cols]
    y = df["target"]
    assert X.isnull().sum().sum() == 0

    return df, X, y, le_target


def build_single_row_features(raw_input, categorical_cols, reference_columns, ordinal_maps=None):
    """Helper function to format a single user input row from Streamlit 
    into the exact encoded format the trained model expects."""
    ordinal_maps = ordinal_maps or ORDINAL_MAPS
    row = pd.DataFrame([raw_input])

    for col, mapping in ordinal_maps.items():
        if col in row.columns:
            row[col] = row[col].map(mapping)

    binary_cols = [c for c in categorical_cols if c not in ordinal_maps and c in row.columns]
    row = pd.get_dummies(row, columns=binary_cols, drop_first=False)

    row = row.reindex(columns=reference_columns, fill_value=0)
    return row


def run_pipeline(path="heart_disease.csv"):
    """Runs the full preprocessing pipeline end to end and returns everything
    downstream code (modeling, app.py) needs.
    """
    df = load_data(path)
    numeric_cols, categorical_cols = get_column_groups(df)
    df, missing_treatment_summary = handle_missing_values(df, numeric_cols, categorical_cols)
    df, X, y, le_target = encode_features(df, categorical_cols, verbose=False)

    return {
        "df": df,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_treatment_summary": missing_treatment_summary,
        "X": X,
        "y": y,
        "le_target": le_target,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("Feature matrix shape:", result["X"].shape)
    print(result["missing_treatment_summary"])