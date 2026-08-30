# Heart Disease Risk Prediction — BMDS2003 Data Science Project

## Overview
This project applies the Cross-Industry Standard Process for Data Mining (CRISP-DM)
framework to a 10,000-record public Heart Disease dataset to explore whether 
patient vitals and lifestyle factors can predict heart disease status. Four models 
were built and compared: Logistic Regression (baseline), Decision Tree, Random Forest, and KNN.

**Key finding:** none of the four models perform meaningfully better than
chance (ROC-AUC ≈ 0.49–0.52 across the board). The dataset does not contain
enough real signal linking the given features to the target for reliable
prediction — this negative result, and why it holds up under multiple
robustness checks, is the main analytical contribution of the report.

## Live demo

🔗 **Streamlit app:** (link)

You can also run the prototype locally — see below.

## Repository contents

| File | Purpose |
|---|---|
| `data_preprocessing.py` | Loading, cleaning, missing-value imputation, encoding |
| `data_visualization.py` | EDA plots, association tests, correlation checks |
| `logistic_regression.py` | Baseline model — training, tuning, evaluation |
| `decision_tree.py` | Decision Tree model — training, tuning, evaluation |
| `random_forest.py` | Random Forest model — training, tuning, evaluation |
| `knn.py` | KNN model — training, tuning, evaluation |
| `feature_selection_check.py` | ANOVA top-10 feature robustness check |
| `pca_check.py` | PCA dimensionality-reduction robustness check |
| `robustness.py` | Alternative-scoring / threshold-tuning / widened-grid checks |
| `find_demo_cases.py` | Utility to search trained models for demo input examples |
| `app.py` | Streamlit prototype — live predictor + full analysis dashboard |
| `heart_3d.py` | 3D heart visualization component used by the app |
| `risk_gauge.py` | Risk gauge visualization component used by the app |
| `heart_disease.csv` | Dataset |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

## Run the prototype locally

```bash
streamlit run app.py
```

The app expects `heart_disease.csv` and the modules above to be in the same
folder it's launched from.

## Notes

- This is a class-project prototype, not a medical diagnostic tool. The app
  itself displays this disclaimer, along with an explanation of *why* each
  prediction was made and a reminder that the model is close to a coin flip.
