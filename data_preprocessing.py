import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TARGET_COL = "Heart Disease Status"
ALCOHOL_COL = "Alcohol Consumption"

ORDINAL_MAPS = {
    "Exercise Habits": {"Low": 0, "Medium": 1, "High": 2},
    "Stress Level": {"Low": 0, "Medium": 1, "High": 2},
    "Sugar Consumption": {"Low": 0, "Medium": 1, "High": 2},
    "Alcohol Consumption": {"Unknown": 0, "Low": 1, "Medium": 2, "High": 3},
}


def load_data(path="heart_disease.csv"):
    df = pd.read_csv(path)

    str_cols = df.select_dtypes(include="object").columns.tolist()
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip().replace("nan", pd.NA)

    df = df.drop_duplicates()
    return df


def get_column_groups(df, target_col=TARGET_COL):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != target_col]
    return numeric_cols, categorical_cols


def handle_missing_values(df, numeric_cols, categorical_cols, alcohol_col=ALCOHOL_COL):
    """Alcohol Consumption is missing completely at random (MCAR) at ~26% --
    too much to mode-impute without flattening real variance, so it's kept
    as its own explicit category instead. Everything else is imputed
    (median for numeric, mode for the remaining categoricals).
    """
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
    """Ordinal columns get an integer mapping that preserves order; the
    remaining binary/nominal columns and the target get label-encoded.
    """
    df = df.copy()
    ordinal_maps = ordinal_maps or ORDINAL_MAPS

    for col, mapping in ordinal_maps.items():
        df[col] = df[col].map(mapping)

    binary_cols = [c for c in categorical_cols if c not in ordinal_maps]
    for col in binary_cols:
        le_col = LabelEncoder()
        df[col] = le_col.fit_transform(df[col])
        if verbose:
            print(col, "->", dict(zip(le_col.classes_, le_col.transform(le_col.classes_))))

    le_target = LabelEncoder()
    df["target"] = le_target.fit_transform(df[target_col])
    if verbose:
        print("target mapping:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

    feature_cols = [c for c in df.columns if c not in [target_col, "target"]]
    X = df[feature_cols]
    y = df["target"]
    assert X.isnull().sum().sum() == 0

    return df, X, y, le_target


def split_and_scale(X, y, test_size=0.20, random_state=RANDOM_STATE):
    """80/20 stratified split, then standardise (fit on train only, applied
    to test) since SVM's distance-based decision boundary would otherwise be
    dominated by features on larger scales (e.g. Cholesterol vs Sleep Hours).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def run_pipeline(path="heart_disease.csv"):
    """Runs the full preprocessing pipeline end to end and returns everything
    downstream code (modeling, app.py) needs.
    """
    df = load_data(path)
    numeric_cols, categorical_cols = get_column_groups(df)
    df, missing_treatment_summary = handle_missing_values(df, numeric_cols, categorical_cols)
    df, X, y, le_target = encode_features(df, categorical_cols, verbose=False)
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_and_scale(X, y)

    return {
        "df": df,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_treatment_summary": missing_treatment_summary,
        "X": X,
        "y": y,
        "le_target": le_target,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("Feature matrix shape:", result["X"].shape)
    print("Train:", result["X_train_scaled"].shape, " Test:", result["X_test_scaled"].shape)
    print(result["missing_treatment_summary"])
