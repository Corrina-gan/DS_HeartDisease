import hashlib
import io
import os
import textwrap
from datetime import datetime

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
import heart_3d
import risk_gauge

st.set_page_config(
    page_title="Heart Disease Risk",
    layout="wide",
    page_icon=":material/favorite:",
)  # light clinical dashboard

st.html(
    """
    <style>
    :root {
        --accent: #0a5c56;
        --accent-dark: #064440;
        --accent-soft: #d7efec;
        --pulse: #e11d48;
        --ink: #16232b;
        --muted: #64748b;
        --line: #d5e3e0;
        --card-radius: 16px;
        --glass: rgba(255, 255, 255, 0.72);
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    input, textarea, select, button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    [data-testid="stAppViewContainer"] {
        background: #f4f8f8;
    }
    [data-testid="stAppViewContainer"]::before { content: none; }
    [data-testid="stHeader"] { background: rgba(244, 248, 248, 0.92); backdrop-filter: blur(12px); }
    .block-container { padding-top: 1.35rem; max-width: 1480px; perspective: 1400px; }

    [data-testid="stDecoration"] {
        background: linear-gradient(90deg, var(--accent) 0%, #14b8a6 100%) !important;
        height: 4px !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fbfa 0%, #eef5f4 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: var(--ink) !important;
        border-bottom: none !important;
        display: block !important;
    }
    [data-testid="stSidebar"] hr { border-color: var(--line); }
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
    [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        display: flex;
        align-items: center;
        width: 100%;
        padding: 10px 14px;
        margin-bottom: 6px;
        border-radius: 12px;
        background-color: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(213, 227, 224, 0.9);
        transition: background 0.18s ease, box-shadow 0.18s ease;
        cursor: pointer;
        font-weight: 500;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: var(--accent-soft) !important;
        box-shadow: 0 6px 16px rgba(15, 118, 110, 0.08);
    }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        display: flex;
        flex-direction: column;
        min-height: calc(100vh - 1.5rem);
    }
    .st-key-sidebar_footer {
        margin-top: auto !important;
        padding-top: 1.1rem !important;
        border-top: 1px solid var(--line);
    }
    .st-key-sidebar_footer [data-testid="stCaptionContainer"] p {
        color: #475569 !important;
        font-size: 0.78rem !important;
        line-height: 1.45 !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: #0a5c56 !important;
        color: #ffffff !important;
        font-weight: 700;
        border-color: #064440;
        box-shadow: 0 8px 18px rgba(6, 68, 64, 0.28);
    }

    h1, h2, h3 {
        color: var(--ink) !important;
        letter-spacing: -0.02em;
        font-weight: 800 !important;
    }
    h1 {
        font-size: 2.35rem !important;
        border-bottom: 3px solid var(--accent);
        padding-bottom: 0.45rem;
        display: block;
        width: 100%;
        margin-bottom: 1.45rem !important;
    }
    h2 { font-size: 1.4rem !important; margin-top: 2.2rem !important; }
    h3 { font-size: 1.1rem !important; margin-top: 1.4rem !important; }

    [data-testid="stMetric"] {
        background: var(--glass);
        border: 1px solid rgba(213, 227, 224, 0.9);
        border-radius: var(--card-radius);
        padding: 0.9rem 1rem 0.7rem 1rem;
        box-shadow: 0 10px 30px rgba(16, 24, 40, 0.05);
        backdrop-filter: blur(10px);
        transform: translateZ(0);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: perspective(900px) rotateX(4deg) translateY(-3px);
        box-shadow: 0 16px 36px rgba(15, 118, 110, 0.12);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.72rem !important;
    }

    [data-testid="stTab"] { font-weight: 600; }
    [data-testid="stTab"][data-selected] { color: #1e3a4c !important; }
    [data-testid="stTab"][data-selected] .react-aria-SelectionIndicator {
        background-color: #0f766e !important;
    }

    @media (prefers-reduced-motion: reduce) {
        [data-testid="stMetric"]:hover,
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            transform: none;
        }
    }

    /* Equal-width split tabs (model / EDA / preprocessing / robustness category bars) */
    .st-key-eda_category_tabs [role="tablist"],
    .st-key-smote_model_tabs [role="tablist"],
    .st-key-prep_category_tabs [role="tablist"],
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
    .st-key-prep_category_tabs [data-testid="stTab"],
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
    .st-key-prep_category_tabs [data-testid="stTab"]:last-of-type,
    .st-key-rob_category_tabs [data-testid="stTab"]:last-of-type {
        border-right: none !important;
    }
    .st-key-eda_category_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"],
    .st-key-smote_model_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"],
    .st-key-prep_category_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"],
    .st-key-rob_category_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"],
    .st-key-eda_category_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"] p,
    .st-key-smote_model_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"] p,
    .st-key-prep_category_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"] p,
    .st-key-rob_category_tabs [data-testid="stTab"] [data-testid="stMarkdownContainer"] p,
    .st-key-eda_category_tabs [data-testid="stTab"] [data-testid="stIconMaterial"],
    .st-key-smote_model_tabs [data-testid="stTab"] [data-testid="stIconMaterial"],
    .st-key-prep_category_tabs [data-testid="stTab"] [data-testid="stIconMaterial"],
    .st-key-rob_category_tabs [data-testid="stTab"] [data-testid="stIconMaterial"] {
        font-size: inherit !important;
        font-weight: 700 !important;
        color: inherit !important;
        text-align: center;
        width: 100%;
    }
    .st-key-eda_category_tabs [data-testid="stTab"][data-hovered]:not([data-selected]),
    .st-key-smote_model_tabs [data-testid="stTab"][data-hovered]:not([data-selected]),
    .st-key-prep_category_tabs [data-testid="stTab"][data-hovered]:not([data-selected]),
    .st-key-rob_category_tabs [data-testid="stTab"][data-hovered]:not([data-selected]) {
        background: #e8edf2 !important;
        color: #1e3a4c !important;
    }
    .st-key-eda_category_tabs [data-testid="stTab"][data-selected],
    .st-key-smote_model_tabs [data-testid="stTab"][data-selected],
    .st-key-prep_category_tabs [data-testid="stTab"][data-selected],
    .st-key-rob_category_tabs [data-testid="stTab"][data-selected],
    .st-key-eda_category_tabs [data-testid="stTab"][aria-selected="true"],
    .st-key-smote_model_tabs [data-testid="stTab"][aria-selected="true"],
    .st-key-prep_category_tabs [data-testid="stTab"][aria-selected="true"],
    .st-key-rob_category_tabs [data-testid="stTab"][aria-selected="true"] {
        background: #0f766e !important;
        color: #ffffff !important;
    }
    .st-key-eda_category_tabs [data-testid="stTab"] .react-aria-SelectionIndicator,
    .st-key-smote_model_tabs [data-testid="stTab"] .react-aria-SelectionIndicator,
    .st-key-prep_category_tabs [data-testid="stTab"] .react-aria-SelectionIndicator,
    .st-key-rob_category_tabs [data-testid="stTab"] .react-aria-SelectionIndicator {
        display: none !important;
    }

    [data-testid="stExpander"] {
        border-radius: var(--card-radius);
        border: 1px solid var(--line);
        background: var(--glass);
        backdrop-filter: blur(10px);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        transform-style: preserve-3d;
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: var(--card-radius);
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.05);
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 40px rgba(15, 118, 110, 0.10);
    }
    .st-key-hero_heart {
        border-radius: 22px;
        overflow: hidden;
        box-shadow: 0 12px 32px rgba(22, 35, 43, 0.08);
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--line);
        border-left-width: 5px;
        border-left-style: solid;
        backdrop-filter: blur(8px);
    }
    [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) { border-left-color: #d97706; }
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) { border-left-color: var(--accent); }
    [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) { border-left-color: #16a34a; }
    [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) { border-left-color: var(--pulse); }

    [data-testid="stProgress"] > div > div > div { background-color: var(--accent) !important; }

    button[kind="primary"], button[kind="formSubmit"] {
        background: #0a5c56 !important;
        border-color: transparent !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 24px rgba(15, 118, 110, 0.28) !important;
        transform: translateZ(0);
        transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    }
    button[kind="primary"]:hover, button[kind="formSubmit"]:hover {
        background: #064440 !important;
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 14px 28px rgba(15, 118, 110, 0.34) !important;
    }

    [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] { margin-bottom: 0.15rem; }

    .st-key-quick_overview {
        margin: 0.15rem 0 0.85rem 0 !important;
    }
    .st-key-quick_overview > div > div > [data-testid="stElementContainer"]:first-child [data-testid="stMarkdownContainer"] p {
        letter-spacing: 0.08em;
        color: #16232b !important;
        margin-bottom: 0.35rem !important;
    }
    .st-key-quick_overview [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    .st-key-quick_overview [data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
    }
    .st-key-quick_overview [data-testid="stColumn"] > div,
    .st-key-qo_patients, .st-key-qo_no, .st-key-qo_yes, .st-key-qo_models {
        height: 100% !important;
        flex: 1 1 auto !important;
    }
    .st-key-qo_patients, .st-key-qo_no, .st-key-qo_yes, .st-key-qo_models {
        border-radius: 14px !important;
        box-shadow: 0 8px 18px rgba(22, 35, 43, 0.07) !important;
    }
    .st-key-qo_patients {
        background: linear-gradient(180deg, #e7f1f8 0%, #ffffff 55%) !important;
        border: 1px solid #a9c7de !important;
        border-top: 6px solid #457b9d !important;
    }
    .st-key-qo_no {
        background: linear-gradient(180deg, #e5f6f1 0%, #ffffff 55%) !important;
        border: 1px solid #9fd6c9 !important;
        border-top: 6px solid #0f766e !important;
    }
    .st-key-qo_yes {
        background: linear-gradient(180deg, #fde8ea 0%, #ffffff 55%) !important;
        border: 1px solid #f3b4b8 !important;
        border-top: 6px solid #e11d48 !important;
    }
    .st-key-qo_models {
        background: linear-gradient(180deg, #fff4d6 0%, #ffffff 55%) !important;
        border: 1px solid #efd48a !important;
        border-top: 6px solid #d4a017 !important;
    }
    .st-key-quick_overview [data-testid="stMetric"] {
        text-align: left;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.2rem 0.1rem 0.05rem !important;
    }
    .st-key-quick_overview [data-testid="stMetric"]:hover {
        transform: none !important;
        box-shadow: none !important;
    }
    .st-key-quick_overview [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        justify-content: flex-start;
    }
    .st-key-quick_overview [data-testid="stMetricValue"] p,
    .st-key-quick_overview [data-testid="stMetricValue"] .stMarkdownColoredText,
    .st-key-quick_overview [data-testid="stMetricValue"] strong {
        font-weight: 800 !important;
    }
    .st-key-qo_patients [data-testid="stMetricValue"],
    .st-key-qo_patients [data-testid="stMetricValue"] p,
    .st-key-qo_patients [data-testid="stMetricValue"] .stMarkdownColoredText {
        color: #1e4e6b !important;
    }
    .st-key-qo_no [data-testid="stMetricValue"],
    .st-key-qo_no [data-testid="stMetricValue"] p,
    .st-key-qo_no [data-testid="stMetricValue"] .stMarkdownColoredText {
        color: #0b5f4b !important;
    }
    .st-key-qo_yes [data-testid="stMetricValue"],
    .st-key-qo_yes [data-testid="stMetricValue"] p,
    .st-key-qo_yes [data-testid="stMetricValue"] .stMarkdownColoredText {
        color: #9f1239 !important;
    }
    .st-key-qo_models [data-testid="stMetricValue"],
    .st-key-qo_models [data-testid="stMetricValue"] p,
    .st-key-qo_models [data-testid="stMetricValue"] .stMarkdownColoredText {
        color: #8a5a00 !important;
    }
    .st-key-quick_overview [data-testid="stMetricLabel"] {
        justify-content: flex-start;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        color: #64748b !important;
    }
    .st-key-live_console {
        background: #ffffff !important;
        border: 1px solid #c5d0ce !important;
        border-radius: 18px !important;
        padding: 1.15rem 1.4rem 1.3rem !important;
        margin-top: 0.55rem !important;
        box-shadow: 0 10px 28px rgba(22, 35, 43, 0.06) !important;
    }
    .st-key-live_console label [data-testid="stMarkdownContainer"] p {
        white-space: normal !important;
        line-height: 1.3 !important;
        font-size: 0.92rem !important;
    }
    .st-key-live_console h3 {
        text-align: center;
        letter-spacing: 0.06em;
        color: #16232b !important;
        margin-bottom: 0.15rem !important;
    }
    .st-key-live_vitals,
    .st-key-live_history {
        background: #f8fbfb !important;
        border: 1px solid #d5e3e0 !important;
        border-radius: 14px !important;
        padding: 0.7rem 0.85rem 0.85rem !important;
    }
    .st-key-live_stage {
        background: linear-gradient(180deg, #f4f8f8 0%, #eef4f3 100%) !important;
        border: 1px solid #c5d4d1 !important;
        border-radius: 16px !important;
        padding: 0.7rem 0.75rem 0.85rem !important;
        box-shadow: 0 10px 28px rgba(22, 35, 43, 0.06) !important;
    }
    .st-key-heart_risk_pct,
    .st-key-heart_risk_pct_bad,
    .st-key-heart_risk_pct_unsure {
        background: #ffffff !important;
        border: 1px solid #d5e3e0 !important;
        border-radius: 14px !important;
        padding: 0.7rem 0.9rem 0.8rem !important;
        margin-top: 0.35rem !important;
        box-shadow: 0 6px 16px rgba(22, 35, 43, 0.06) !important;
    }
    .st-key-heart_risk_pct [data-testid="stMetricValue"],
    .st-key-heart_risk_pct_bad [data-testid="stMetricValue"],
    .st-key-heart_risk_pct_unsure [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        color: #0a5c56 !important;
    }
    .st-key-heart_risk_pct_bad [data-testid="stMetricValue"] { color: #be123c !important; }
    .st-key-heart_risk_pct_unsure [data-testid="stMetricValue"] { color: #b45309 !important; }
    .st-key-heart_risk_pct [data-testid="stMetricLabel"],
    .st-key-heart_risk_pct_bad [data-testid="stMetricLabel"],
    .st-key-heart_risk_pct_unsure [data-testid="stMetricLabel"] {
        font-weight: 700 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #475569 !important;
    }
    .st-key-assess_ready,
    .st-key-assess_bad,
    .st-key-assess_ok,
    .st-key-assess_unsure {
        background: #ffffff !important;
        border-radius: 0 12px 12px 0 !important;
        padding: 0.55rem 0.75rem 0.65rem 0.9rem !important;
        box-shadow: none !important;
    }
    .st-key-assess_ready { border-left: 3px solid #94a3b8 !important; border-top: none !important; border-right: none !important; border-bottom: none !important; }
    .st-key-assess_bad { border-left: 3px solid #e11d48 !important; border-top: none !important; border-right: none !important; border-bottom: none !important; }
    .st-key-assess_ok { border-left: 3px solid #0f766e !important; border-top: none !important; border-right: none !important; border-bottom: none !important; }
    .st-key-assess_unsure { border-left: 3px solid #ca8a04 !important; border-top: none !important; border-right: none !important; border-bottom: none !important; }
    .st-key-explain_panel_ok, .st-key-explain_panel_bad, .st-key-explain_panel_unsure {
        background: #ffffff !important;
        border: 1px solid #d5e3e0 !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 26px rgba(22, 35, 43, 0.07) !important;
        padding: 0.35rem 0.25rem 0.2rem !important;
    }
    .st-key-explain_panel_ok { border-top: 6px solid #0a5c56 !important; }
    .st-key-explain_panel_bad { border-top: 6px solid #e11d48 !important; }
    .st-key-explain_panel_unsure { border-top: 6px solid #d4a017 !important; }
    .st-key-explain_panel_ok [data-testid="stMarkdownContainer"] p,
    .st-key-explain_panel_bad [data-testid="stMarkdownContainer"] p,
    .st-key-explain_panel_unsure [data-testid="stMarkdownContainer"] p {
        line-height: 1.55 !important;
    }
    .st-key-explain_panel_ok [data-testid="stMarkdownContainer"] strong,
    .st-key-explain_panel_bad [data-testid="stMarkdownContainer"] strong,
    .st-key-explain_panel_unsure [data-testid="stMarkdownContainer"] strong {
        letter-spacing: 0.01em;
    }
    .st-key-cta_row {
        margin: 0.55rem 0 0.35rem 0 !important;
    }
    .st-key-analyze_heart button {
        letter-spacing: 0.06em;
        font-weight: 800 !important;
        min-height: 3rem !important;
        font-size: 1.02rem !important;
    }
    .st-key-reset_form button {
        min-height: 3rem !important;
        font-weight: 700 !important;
    }
    .st-key-visit_history {
        margin-top: 0.65rem !important;
        border-top: 6px solid #0a5c56 !important;
    }
    .st-key-export_assessment_pdf button {
        min-height: 2.75rem !important;
        font-weight: 700 !important;
    }
    .st-key-explore_further {
        margin-top: 0.35rem !important;
    }
    .st-key-explore_further [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    .st-key-explore_further [data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
    }
    .st-key-explore_further [data-testid="stColumn"] > div,
    .st-key-explore_eda, .st-key-explore_models {
        height: 100% !important;
        flex: 1 1 auto !important;
    }
    .st-key-explore_eda, .st-key-explore_models {
        border-radius: 16px !important;
        box-shadow: 0 10px 24px rgba(22, 35, 43, 0.07) !important;
    }
    .st-key-explore_eda {
        background: linear-gradient(180deg, #e5f6f1 0%, #ffffff 48%) !important;
        border: 1px solid #9fd6c9 !important;
        border-top: 6px solid #0f766e !important;
    }
    .st-key-explore_models {
        background: linear-gradient(180deg, #e7f1f8 0%, #ffffff 48%) !important;
        border: 1px solid #a9c7de !important;
        border-top: 6px solid #457b9d !important;
    }
    .st-key-explore_eda h3, .st-key-explore_models h3 {
        margin: 0.15rem 0 0.35rem 0 !important;
        font-size: 1.2rem !important;
        letter-spacing: -0.02em;
    }
    .st-key-explore_eda [data-testid="stCaptionContainer"] p,
    .st-key-explore_models [data-testid="stCaptionContainer"] p {
        color: #475569 !important;
        line-height: 1.5 !important;
        font-size: 0.95rem !important;
    }
    .st-key-explore_eda button {
        background: linear-gradient(135deg, #14b8a6 0%, #0f766e 100%) !important;
        border-color: transparent !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 18px rgba(15, 118, 110, 0.22) !important;
    }
    .st-key-explore_models button {
        background: linear-gradient(135deg, #6ea3c4 0%, #457b9d 100%) !important;
        border-color: transparent !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 18px rgba(69, 123, 157, 0.22) !important;
    }
    </style>
    """
)

PULSE_DIVIDER_SVG = """
<svg width="260" height="22" viewBox="0 0 260 22" xmlns="http://www.w3.org/2000/svg" style="margin: 2px 0 10px 0;">
  <defs>
    <linearGradient id="ecgGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#0f766e"/>
      <stop offset="55%" stop-color="#14b8a6"/>
      <stop offset="100%" stop-color="#e11d48"/>
    </linearGradient>
  </defs>
  <polyline class="ecg-line" points="0,11 70,11 82,3 94,19 106,11 120,11 128,5 136,17 144,11 260,11"
    fill="none" stroke="url(#ecgGrad)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
  <style>
    .ecg-line { stroke-dasharray: 90 420; animation: ecgSweep 2.4s linear infinite; }
    @keyframes ecgSweep { to { stroke-dashoffset: -510; } }
    @media (prefers-reduced-motion: reduce) { .ecg-line { animation: none; stroke-dasharray: none; } }
  </style>
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


def _best_param(metrics, prefix, name):
    params = metrics.get("Best Params") or {}
    return params.get(f"{prefix}__{name}", params.get(name))


def _hp_display(value, kind="plain"):
    if value is None or value == "None":
        return "None"
    if kind == "penalty":
        return str(value).upper()
    if kind == "weight":
        return str(value).replace("_", " ").title()
    if kind == "metric":
        return str(value).replace("_", " ").title()
    return value


def render_basic_vs_smote_eval(model_name, basic_m, smote_m, hyperparams, tune_what):
    """Final hyperparameter table (Basic vs SMOTE) plus what each test metric means."""
    setting_names = [label for label, *_ in hyperparams]
    basic_hp = [_hp_display(_best_param(basic_m, prefix, key), kind) for _, prefix, key, kind in hyperparams]
    smote_hp = [_hp_display(_best_param(smote_m, prefix, key), kind) for _, prefix, key, kind in hyperparams]
    metric_names = ["CV F1", "Test Accuracy", "Test Recall", "Test F1", "Test ROC-AUC"]
    metric_keys = ["CV F1 (best)", "Accuracy", "Recall", "F1-Score", "ROC-AUC"]

    config = pd.DataFrame({
        "Pipeline": setting_names + metric_names,
        "Basic": basic_hp + [f"{basic_m[k]:.4f}" for k in metric_keys],
        "SMOTE": smote_hp + [f"{smote_m[k]:.4f}" for k in metric_keys],
    })

    kept = "Basic" if basic_m["ROC-AUC"] >= smote_m["ROC-AUC"] else "SMOTE"
    smote_raised_f1 = smote_m["F1-Score"] > basic_m["F1-Score"] + 1e-6
    if kept == "Basic" and smote_raised_f1:
        select_note = (
            "The **Basic** pipeline was selected for comparison (higher test ROC-AUC). "
            "SMOTE improved test F1 but not ranking quality."
        )
        f1_tail = "SMOTE raised this score, but ROC-AUC did not improve, so **Basic** is kept."
    elif kept == "Basic":
        select_note = (
            "The **Basic** pipeline was selected for comparison. "
            "Oversampling did not improve the test F1-score."
        )
        f1_tail = "SMOTE did not raise this score, so the **Basic** pipeline is kept."
    else:
        select_note = "The **SMOTE** pipeline was selected for comparison (higher test ROC-AUC)."
        f1_tail = "SMOTE improved this score and is kept for comparison."

    st.subheader("⚙️ Final Hyperparameter Configuration")

    with st.container(border=True):
        st.markdown(f"**Final configuration — {model_name} (Basic vs SMOTE)**")
        st.dataframe(config, width="stretch", hide_index=True)
        st.caption(select_note)

    with st.container(border=True):
        st.markdown("**What these evaluation metrics mean**")
        st.markdown(
            f"- **CV F1** (Basic {basic_m['CV F1 (best)']:.4f} | SMOTE {smote_m['CV F1 (best)']:.4f}) — "
            "mean F1 for the Yes class during 5-fold tuning on the *training* set. "
            f"GridSearch used this score to pick {tune_what}."
        )
        st.markdown(
            f"- **Test accuracy** (Basic {basic_m['Accuracy']:.4f} | SMOTE {smote_m['Accuracy']:.4f}) — "
            "share of all test patients labelled correctly. "
            "Always predicting No already scores about **0.80**, so accuracy alone is a poor headline metric on this 80/20 set."
        )
        st.markdown(
            f"- **Test precision** (Basic {basic_m['Precision']:.4f} | SMOTE {smote_m['Precision']:.4f}) — "
            "of the patients the model called **Yes**, how many really had heart disease. "
            f"Basic {basic_m['Precision']:.0%} / SMOTE {smote_m['Precision']:.0%} means most Yes flags are false alarms."
        )
        st.markdown(
            f"- **Test recall** (Basic {basic_m['Recall']:.4f} | SMOTE {smote_m['Recall']:.4f}) — "
            "of the patients who truly had heart disease, how many the model found. "
            f"Basic finds **{basic_m['Recall']:.1%}** of real Yes cases; SMOTE finds **{smote_m['Recall']:.1%}**."
        )
        st.markdown(
            f"- **Test F1** (Basic {basic_m['F1-Score']:.4f} | SMOTE {smote_m['F1-Score']:.4f}) — "
            f"harmonic mean of precision and recall for the Yes class. {f1_tail}"
        )
        st.markdown(
            f"- **Test ROC-AUC** (Basic {basic_m['ROC-AUC']:.4f} | SMOTE {smote_m['ROC-AUC']:.4f}) — "
            "how well the model *ranks* Yes above No. **0.50** is a coin flip. "
            "Values near chance mean neither pipeline is a reliable risk ranker."
        )


def _pipeline_input_for_clf(pipeline, row):
    """Apply every transform before the classifier (scaler yes, SMOTE no)."""
    Xt = row
    clf_names = {"knn", "logreg", "dt", "rf"}
    for name, step in pipeline.steps:
        if name in clf_names:
            break
        if name == "smote" or not hasattr(step, "transform"):
            continue
        Xt = step.transform(Xt)
    return np.asarray(Xt, dtype=float)


def _source_field(feature_name, raw_input):
    """Map an encoded column name back to the form field the person filled in."""
    if raw_input and feature_name in raw_input:
        return feature_name, raw_input[feature_name]
    if raw_input:
        matches = [key for key in raw_input if feature_name.startswith(f"{key}_")]
        if matches:
            key = max(matches, key=len)
            return key, raw_input[key]
    return f"{feature_name.replace('_', ' ')}", None


EVIDENCE_FIELDS = [
    "Age",
    "BMI",
    "Blood Pressure",
    "Cholesterol Level",
    "Smoking",
    "Diabetes",
    "Exercise Habits",
    "Family Heart Disease",
]


def _fmt_answer(col, value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if col in ("Age", "Blood Pressure", "Cholesterol Level", "Triglyceride Level", "Fasting Blood Sugar"):
        try:
            return str(int(round(float(value))))
        except (TypeError, ValueError):
            return str(value)
    if col in ("BMI", "Sleep Hours", "CRP Level", "Homocysteine Level"):
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


@st.cache_data(show_spinner=False)
def typical_answers_by_group(_raw_df):
    """Typical (median / most common) answer in the No group vs the Yes group."""
    raw_df = _raw_df
    target = dp.TARGET_COL
    no = raw_df[raw_df[target] == "No"]
    yes = raw_df[raw_df[target] == "Yes"]
    out = {}
    for col in raw_df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(raw_df[col]):
            out[col] = {"kind": "num", "no": no[col].median(), "yes": yes[col].median()}
        else:
            no_mode = no[col].dropna().mode()
            yes_mode = yes[col].dropna().mode()
            out[col] = {
                "kind": "cat",
                "no": None if no_mode.empty else no_mode.iloc[0],
                "yes": None if yes_mode.empty else yes_mode.iloc[0],
            }
    return out


def build_evidence_table(raw_input, raw_df, highlight_fields=None):
    """Side-by-side: your answers vs typical Yes/No people in the dataset."""
    typicals = typical_answers_by_group(raw_df)
    ordered = []
    seen = set()
    for field in list(highlight_fields or []) + EVIDENCE_FIELDS:
        if field in raw_input and field in typicals and field not in seen:
            ordered.append(field)
            seen.add(field)
        if len(ordered) >= 8:
            break

    rows = []
    similar_count = 0
    for field in ordered:
        spec = typicals[field]
        you = _fmt_answer(field, raw_input.get(field))
        no_s = _fmt_answer(field, spec["no"])
        yes_s = _fmt_answer(field, spec["yes"])
        if no_s == yes_s:
            reading = "Looks similar in both groups"
            similar_count += 1
        elif spec["kind"] == "num":
            try:
                you_n, no_n, yes_n = float(you), float(no_s), float(yes_s)
                reading = (
                    "Closer to people without heart disease"
                    if abs(you_n - no_n) < abs(you_n - yes_n)
                    else "Closer to people with heart disease"
                )
            except ValueError:
                reading = "See the numbers"
        else:
            if you == no_s and you != yes_s:
                reading = "More common if no heart disease"
            elif you == yes_s and you != no_s:
                reading = "More common if heart disease"
            else:
                reading = "Common in both groups"
                similar_count += 1
        rows.append({
            "Question": field,
            "Your answer": you,
            "Usually without heart disease": no_s,
            "Usually with heart disease": yes_s,
            "What this suggests": reading,
        })
    return pd.DataFrame(rows), similar_count, len(rows)


def _reason_bullets(evidence_df, max_bullets=3):
    """Turn the evidence table into 2-3 short, scannable, plain-English lines —
    only the rows that actually lean toward Yes or No. Rows where the answer
    looked the same in both groups are skipped here (they don't explain
    anything) but are still visible in the full table for anyone who wants
    to check every row."""
    skip = {"Looks similar in both groups", "Common in both groups", "See the numbers"}
    bullets = []
    for _, row in evidence_df.iterrows():
        suggestion = row["What this suggests"]
        if suggestion in skip:
            continue
        bullets.append(f"**{row['Question']}** ({row['Your answer']}) — {suggestion[0].lower()}{suggestion[1:]}")
        if len(bullets) >= max_bullets:
            break
    return bullets


def _original_fields(encoded_names, raw_input):
    fields = []
    seen = set()
    for name in encoded_names:
        field, _ = _source_field(name, raw_input)
        if raw_input and field in raw_input and field not in seen:
            seen.add(field)
            fields.append(field)
    return fields


def verdict_tier(pct):
    """The one place that decides Unlikely / Unclear / More likely from a
    score. Every verdict shown on screen (sidebar badge, gauge status,
    explanation panel) calls this instead of each doing its own
    Yes/No-vs-score check independently — that's what previously let the
    sidebar say "LOWER RISK" (from the binary label) while the panel below
    it said "Unclear" (from the score) for the exact same prediction.
    """
    if pct < 40:
        return {"word": "Unlikely", "icon": "🟢", "css": "ok", "badge": "LOWER RISK"}
    if pct > 60:
        return {"word": "More likely", "icon": "🔴", "css": "bad", "badge": "HIGHER RISK"}
    return {"word": "Unclear", "icon": "🟡", "css": "unsure", "badge": "UNCLEAR"}


def explain_live_prediction(pipeline, row, pred, prob_disease, raw_input=None):
    """Short, non-technical reason for the live predictor.

    Rewritten so someone with no data-science background can follow it:
    a one-line verdict, what the percentage actually means, a plain-English
    "how it decided" line for whichever model is active, and a clear
    reminder that this project found the model close to a coin flip — so a
    low score is reassuring, not a guarantee.
    """
    pct = prob_disease * 100
    tier = verdict_tier(pct)

    meaning = (
        "This isn't measuring disease directly — it's comparing this form against patterns "
        "found in 10,000 past records, then turning that comparison into a score out of 100. "
        "**50 or above** becomes a \"Yes\"; below that is \"No\"."
    )

    why = ""
    highlight_fields = []
    names = list(row.columns)
    Xt = _pipeline_input_for_clf(pipeline, row)
    x = Xt[0]
    steps = pipeline.named_steps

    if "knn" in steps:
        knn = steps["knn"]
        k = int(knn.n_neighbors)
        _, idx = knn.kneighbors(Xt, n_neighbors=k)
        neighbor_y = np.asarray(knn._y)[idx[0]]
        n_yes = int(neighbor_y.sum())
        n_no = k - n_yes
        why = (
            f"**How it decided:** it found the **{k} people in our records who answered most like this**, "
            f"almost like looking up {k} similar patients. "
            f"**{n_no} of {k}** did **not** have heart disease"
            + (f" and **{n_yes}** did." if n_yes else ".")
            + " It went with whichever group was bigger. The table below shows the answers it used to judge \"similar\"."
        )
        highlight_fields = list(EVIDENCE_FIELDS)
    elif "logreg" in steps:
        logreg = steps["logreg"]
        contrib = logreg.coef_[0] * x
        order = np.argsort(-np.abs(contrib))
        ranked = [names[i] for i in order if abs(contrib[i]) >= 1e-9]
        why = (
            "**How it decided:** picture a checklist where every answer either adds a few points "
            "or takes a few away. It added those points up to get the score above. "
            "The rows at the top of the table below moved this person's score the most — "
            "the rest barely moved it at all."
        )
        highlight_fields = _original_fields(ranked, raw_input)
    elif "dt" in steps:
        tree = steps["dt"].tree_
        node = 0
        asked = []
        while tree.children_left[node] != -1:
            asked.append(names[int(tree.feature[node])])
            feat_i = int(tree.feature[node])
            if float(x[feat_i]) <= float(tree.threshold[node]):
                node = int(tree.children_left[node])
            else:
                node = int(tree.children_right[node])
        why = (
            "**How it decided:** it worked through a short series of yes/no questions about the "
            "form, like a flowchart, and ended up with a group of similar past patients. "
            "Whatever that group mostly was — Yes or No — became the answer. "
            "Those questions are listed first in the table below."
        )
        highlight_fields = _original_fields(asked, raw_input)
    elif "rf" in steps:
        rf = steps["rf"]
        tree_votes = np.array([est.predict(Xt)[0] for est in rf.estimators_])
        n_yes = int(tree_votes.sum())
        n_trees = len(tree_votes)
        why = (
            f"**How it decided:** it's really **{n_trees} small decision-makers voting**, each one "
            "looking at the form slightly differently, then going with the majority. "
            f"**{n_trees - n_yes}** voted No and **{n_yes}** voted Yes. "
            "The table below shows the answers it usually pays the most attention to."
        )
        top = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False).index.tolist()
        highlight_fields = _original_fields(top, raw_input)

    note = (
        "**About this tool:** this is a **class project demo** built on 10,000 public records, "
        "not a real diagnostic test. Testing in this project found the model's predictions were "
        "barely better than a random guess, so please treat any score here as a talking point, "
        "not a result — and speak to a doctor about any real health concerns."
    )
    return {
        "tier": tier,
        "meaning": meaning,
        "why": why,
        "highlight_fields": highlight_fields,
        "note": note,
    }


@st.cache_data(show_spinner="Loading & preprocessing data...")
def load_pipeline_data(path):
    return dp.run_pipeline(path)


@st.cache_data(show_spinner="Loading raw data...")
def load_raw_data(path):
    df = dp.load_data(path)
    numeric_cols, categorical_cols = dp.get_column_groups(df)
    return df, numeric_cols, categorical_cols


@st.cache_resource(show_spinner=False)
def train_knn_basic(_X, _y, cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = km.get_70_30_split(X, y)
        result = km.tune_and_evaluate(
            km.build_basic_pipeline(), km.BASIC_PARAM_GRID,
            X_train, X_test, y_train, y_test, "1. Basic KNN",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("knn_basic", _compute, _X, _y)


@st.cache_resource(show_spinner=False)
def train_knn_smote(_X, _y, cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = km.get_70_30_split(X, y)
        result = km.tune_and_evaluate(
            km.build_smote_pipeline(), km.SMOTE_PARAM_GRID,
            X_train, X_test, y_train, y_test, "2. SMOTE KNN",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("knn_smote", _compute, _X, _y)


@st.cache_resource(show_spinner=False)
def train_dt_basic(_X, _y, cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = dtm.get_70_30_split(X, y)
        result = dtm.tune_and_evaluate(
            dtm.build_basic_pipeline(), dtm.BASIC_PARAM_GRID,
            X_train, X_test, y_train, y_test, "1. Basic Decision Tree",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("dt_basic", _compute, _X, _y)


@st.cache_resource(show_spinner=False)
def train_dt_smote(_X, _y, cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = dtm.get_70_30_split(X, y)
        result = dtm.tune_and_evaluate(
            dtm.build_smote_pipeline(), dtm.SMOTE_PARAM_GRID,
            X_train, X_test, y_train, y_test, "2. SMOTE Decision Tree",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("dt_smote", _compute, _X, _y)


@st.cache_resource(show_spinner=False)
def train_lr_basic(_X, _y, cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = lgm.get_70_30_split(X, y)
        result = lgm.tune_and_evaluate(
            lgm.build_basic_pipeline(), lgm.BASIC_PARAM_GRID,
            X_train, X_test, y_train, y_test, "1. Basic Logistic Regression",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("lr_basic", _compute, _X, _y)


@st.cache_resource(show_spinner=False)
def train_lr_smote(_X, _y, cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = lgm.get_70_30_split(X, y)
        result = lgm.tune_and_evaluate(
            lgm.build_smote_pipeline(), lgm.SMOTE_PARAM_GRID,
            X_train, X_test, y_train, y_test, "2. SMOTE Logistic Regression",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("lr_smote", _compute, _X, _y)


@st.cache_resource(show_spinner=False)
def train_rf_basic(_X, _y, cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = rfm.get_70_30_split(X, y)
        result = rfm.tune_and_evaluate(
            rfm.build_basic_pipeline(), rfm.BASIC_PARAM_GRID,
            X_train, X_test, y_train, y_test, "1. Basic Random Forest",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("rf_basic", _compute, _X, _y)


@st.cache_resource(show_spinner=False)
def train_rf_smote(_X, _y, cache_tag=FEATURE_CACHE_TAG):
    def _compute(X, y):
        X_train, X_test, y_train, y_test = rfm.get_70_30_split(X, y)
        result = rfm.tune_and_evaluate(
            rfm.build_smote_pipeline(), rfm.SMOTE_PARAM_GRID,
            X_train, X_test, y_train, y_test, "2. SMOTE Random Forest",
        )
        return {"X_test": X_test, "y_test": y_test, "result": result}
    return load_or_train("rf_smote", _compute, _X, _y)

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
    progress.empty()
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

    progress.empty()

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


MODEL_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN"]


def plot_selected_pipelines_roc_auc(basic_results, y_tests):
    """Figure 6.2 — ROC-AUC of the four Basic pipelines, recomputed from test probabilities."""
    order = MODEL_ORDER
    aucs = [roc_auc_score(y_tests[name], basic_results[name]["y_prob"]) for name in order]
    labels = [f"Basic\n{name}" for name in order]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    bars = ax.bar(labels, aucs, color=colors, width=0.62, zorder=3)
    ax.axhline(0.5, color="#c44e52", linestyle="--", linewidth=1.6, label="Chance level (0.500)", zorder=2)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in aucs], padding=4, fontsize=10, fontweight="bold")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 0.6)
    ax.set_title("ROC-AUC of the 4 Selected (Basic) Pipelines", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def get_feature_selection_result(_X, _y):
    """ANOVA top-10 feature selection robustness check (report Section 5.6.5)."""
    return load_or_compute(
        "feature_selection_anova",
        lambda: fscm.run_feature_selection_check(data={"X": _X, "y": _y}, save_outputs=False),
    )


@st.cache_resource(show_spinner=False)
def get_pca_result(_X, _y):
    """PCA dimensionality-reduction robustness check (report Section 5.6.6)."""
    return load_or_compute(
        "pca_robustness",
        lambda: pcam.run_pca_check(data={"X": _X, "y": _y}, save_outputs=False),
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

    st.markdown(
        "<h2 style='text-align:center;margin-bottom:0.2rem;'>🫀 Heart Disease Risk</h2>",
        unsafe_allow_html=True,
    )
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

    with st.container(key="sidebar_footer"):
        st.caption("10,000-record public dataset · 4 models")

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

    _basic = {
        "KNN": knn_basic_data["result"],
        "Logistic Regression": lr_basic_data["result"],
        "Random Forest": rf_basic_data["result"],
        "Decision Tree": dt_basic_data["result"],
    }
    _smote = {
        "KNN": knn_smote_data["result"],
        "Logistic Regression": lr_smote_data["result"],
        "Random Forest": rf_smote_data["result"],
        "Decision Tree": dt_smote_data["result"],
    }
    basic_results = {name: _basic[name] for name in MODEL_ORDER}
    smote_results = {name: _smote[name] for name in MODEL_ORDER}

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


@st.cache_resource(show_spinner=False)
def get_patient_cloud_fig(_raw_df, target_col):
    # heart_3d.plot_patient_cloud_3d rebuilds a 1,600-point 3D scatter from
    # scratch every call; it was previously invoked unmemoized directly in
    # the render path, so it re-ran on every single widget interaction that
    # triggers a Streamlit rerun (not just when the EDA page is first
    # opened). Caching it here means it is only built once per (raw_df,
    # target_col) pair.
    return heart_3d.plot_patient_cloud_3d(_raw_df, target_col)


@st.cache_resource(show_spinner="Prefetching EDA Plots (One-time setup)...")
def prefetch_eda_plots(_df, num_cols, cat_cols):
    df = _df
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
        "cat_pct": dv.plot_categorical_percentage_by_target(df, cat_cols),
    }

@st.cache_resource(show_spinner="Prefetching Data Stats...")
def prefetch_stats(_df, num_cols, cat_cols, _X_df, _y_ser):
    df, X_df, y_ser = _df, _X_df, _y_ser
    outlier_df = dv.compute_outlier_counts(df, num_cols)
    table_numeric, table_categorical, fig_assoc = dv.test_target_associations(df, num_cols, cat_cols)
    mcar_df, fig_mcar = dv.test_alcohol_missingness_mcar(df, num_cols, cat_cols)
    fig_corr, _ = dv.plot_target_correlation_heatmap(pipeline_data["df"], X_df.columns.tolist())
    anova_df = dv.compute_anova_scores(X_df, y_ser)
    chi2_df = dv.compute_chi2_scores(X_df, y_ser)
    return outlier_df, table_numeric, table_categorical, fig_assoc, mcar_df, fig_mcar, fig_corr, anova_df, chi2_df


FEATURED_NUM = [
    "Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours",
]
FEATURED_CAT = [
    "Gender", "Exercise Habits", "Smoking",
    "Family Heart Disease", "Diabetes", "High Blood Pressure",
]
FIELD_UNITS = {
    "Age": "years",
    "Blood Pressure": "mmHg",
    "Cholesterol Level": "mg/dL",
    "BMI": "kg/m²",
    "Sleep Hours": "h",
    "Triglyceride Level": "mg/dL",
    "Fasting Blood Sugar": "mg/dL",
    "CRP Level": "mg/L",
    "Homocysteine Level": "µmol/L",
}
SHORT_LABELS = {
    "Age": "Age",
    "Blood Pressure": "Blood Pressure",
    "Cholesterol Level": "Cholesterol Level",
    "BMI": "Body Mass Index",
    "Sleep Hours": "Sleep Hours",
    "Fasting Blood Sugar": "Fasting Blood Sugar",
    "CRP Level": "C-Reactive Protein",
    "Triglyceride Level": "Triglyceride Level",
    "Homocysteine Level": "Homocysteine Level",
    "Gender": "Gender",
    "Exercise Habits": "Exercise Habits",
    "Smoking": "Smoking",
    "Family Heart Disease": "Family Heart Disease",
    "Diabetes": "Diabetes",
    "High Blood Pressure": "High Blood Pressure",
    "Low HDL Cholesterol": "Low HDL (High-Density Lipoprotein)",
    "High LDL Cholesterol": "High LDL (Low-Density Lipoprotein)",
    "Alcohol Consumption": "Alcohol Consumption",
    "Stress Level": "Stress Level",
    "Sugar Consumption": "Sugar Consumption",
}


def _vital_flag(kind, value):
    """Return (dot, label) for live vitals. kind is bp, chol, or bmi."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "⚪", "—"
    if kind == "bp":
        if v >= 140:
            return "🔴", "High"
        if v >= 130:
            return "🟠", "Elevated"
        return "🟢", "Normal"
    if kind == "chol":
        if v >= 240:
            return "🔴", "High"
        if v >= 200:
            return "🟠", "Elevated"
        return "🟢", "Normal"
    if kind == "bmi":
        if v >= 30:
            return "🔴", "High"
        if v >= 25:
            return "🟠", "Elevated"
        return "🟢", "Normal"
    return "⚪", "—"


def _estimated_bpm():
    """Resting-style BPM from form vitals (there is no heart-rate field)."""
    age = float(st.session_state.get("pred_num_Age", 49) or 49)
    bp = float(st.session_state.get("pred_num_Blood Pressure", 120) or 120)
    sleep = float(st.session_state.get("pred_num_Sleep Hours", 7) or 7)
    stress = str(st.session_state.get("pred_cat_Stress Level", "Low") or "Low")
    bpm = 68.0
    if age >= 60:
        bpm += 4
    if bp >= 140:
        bpm += 10
    elif bp >= 130:
        bpm += 5
    if sleep <= 5:
        bpm += 10
    elif sleep <= 6:
        bpm += 5
    if stress == "High":
        bpm += 14
    elif stress == "Medium":
        bpm += 6
    if st.session_state.get("last_risk_label") == "Yes":
        bpm += 8
    return int(min(120, max(56, round(bpm))))


def _field_label(col):
    name = SHORT_LABELS.get(col, col)
    unit = FIELD_UNITS.get(col)
    return f"{name} ({unit})" if unit else name


def _num_widget(col, default):
    label = _field_label(col)
    key = f"pred_num_{col}"
    if key not in st.session_state:
        st.session_state[key] = float(default)
    kwargs = {"key": key, "persist_state": "session"}
    if col in ("BMI", "Sleep Hours", "CRP Level", "Homocysteine Level"):
        kwargs["format"] = "%.2f"
        kwargs["step"] = 0.01 if col == "BMI" else 0.1
    else:
        kwargs["format"] = "%.0f"
        kwargs["step"] = 1.0
    return st.number_input(label, **kwargs)


def _cat_widget(col, options):
    key = f"pred_cat_{col}"
    if key not in st.session_state and options:
        st.session_state[key] = options[0]
    return st.selectbox(
        SHORT_LABELS.get(col, col),
        options,
        key=key,
        persist_state="session",
    )


CUSTOM_PRESET = "Custom (edit the form)"
PATIENT_PRESETS = {
    "Healthy Active Adult": {
        "num": {
            "Age": 34, "Blood Pressure": 112, "Cholesterol Level": 168, "BMI": 22.4,
            "Sleep Hours": 8.0, "Triglyceride Level": 88, "Fasting Blood Sugar": 86,
            "CRP Level": 0.6, "Homocysteine Level": 7.8,
        },
        "cat": {
            "Gender": "Female", "Exercise Habits": "High", "Smoking": "No",
            "Family Heart Disease": "No", "Diabetes": "No", "High Blood Pressure": "No",
            "Low HDL Cholesterol": "No", "High LDL Cholesterol": "No",
            "Alcohol Consumption": "Low", "Stress Level": "Low", "Sugar Consumption": "Low",
        },
    },
    "High-Risk Diabetic": {
        "num": {
            "Age": 80, "Blood Pressure": 200, "Cholesterol Level": 320, "BMI": 42.0,
            "Sleep Hours": 3.0, "Triglyceride Level": 380, "Fasting Blood Sugar": 198,
            "CRP Level": 16.5, "Homocysteine Level": 24.0,
        },
        "cat": {
            "Gender": "Male", "Exercise Habits": "Low", "Smoking": "Yes",
            "Family Heart Disease": "Yes", "Diabetes": "Yes", "High Blood Pressure": "Yes",
            "Low HDL Cholesterol": "Yes", "High LDL Cholesterol": "Yes",
            "Alcohol Consumption": "High", "Stress Level": "High", "Sugar Consumption": "High",
        },
    },
    "Adult With Some Risk Factors": {
        "num": {
            "Age": 56, "Blood Pressure": 132, "Cholesterol Level": 212, "BMI": 27.1,
            "Sleep Hours": 6.0, "Triglyceride Level": 178, "Fasting Blood Sugar": 108,
            "CRP Level": 3.1, "Homocysteine Level": 12.8,
        },
        "cat": {
            "Gender": "Male", "Exercise Habits": "Medium", "Smoking": "No",
            "Family Heart Disease": "Yes", "Diabetes": "No", "High Blood Pressure": "No",
            "Low HDL Cholesterol": "No", "High LDL Cholesterol": "Yes",
            "Alcohol Consumption": "Medium", "Stress Level": "Medium", "Sugar Consumption": "Medium",
        },
    },
}


def _form_defaults(raw_df, numeric_cols, categorical_cols):
    nums = {col: float(raw_df[col].median()) for col in numeric_cols}
    cats = {}
    for col in categorical_cols:
        mode = raw_df[col].dropna().mode()
        if not mode.empty:
            cats[col] = mode.iloc[0]
        elif col in dp.ORDINAL_MAPS:
            cats[col] = next(iter(dp.ORDINAL_MAPS[col]))
        else:
            cats[col] = "No"
    return nums, cats


def _apply_profile(nums, cats):
    for col, val in nums.items():
        st.session_state[f"pred_num_{col}"] = float(val)
    for col, val in cats.items():
        st.session_state[f"pred_cat_{col}"] = val


def _seed_form_defaults(raw_df, numeric_cols, categorical_cols):
    nums, cats = _form_defaults(raw_df, numeric_cols, categorical_cols)
    for col, val in nums.items():
        st.session_state.setdefault(f"pred_num_{col}", float(val))
    for col, val in cats.items():
        st.session_state.setdefault(f"pred_cat_{col}", val)


def _on_patient_preset():
    name = st.session_state.get("patient_preset")
    spec = PATIENT_PRESETS.get(name)
    if spec:
        _apply_profile(spec["num"], spec["cat"])


def _reset_patient_form(raw_df, numeric_cols, categorical_cols):
    nums, cats = _form_defaults(raw_df, numeric_cols, categorical_cols)
    _apply_profile(nums, cats)
    st.session_state.patient_preset = CUSTOM_PRESET
    st.session_state.last_risk_pct = None
    st.session_state.last_risk_label = None
    st.session_state.live_pred = None


def _record_visit(prob_disease, label, model_choice):
    hist = st.session_state.setdefault("risk_history", [])
    hist.append({
        "visit": len(hist) + 1,
        "when": datetime.now().strftime("%H:%M"),
        "risk": float(prob_disease) * 100,
        "label": label,
        "model": model_choice,
    })


def build_assessment_pdf(live, history):
    """One-page printable summary. Class-project demo, not a medical record."""
    raw = live.get("raw_input") or {}
    reason = live.get("reason") or {}
    pct = float(live.get("prob_disease") or 0) * 100
    label = live.get("label") or "—"
    model = live.get("model_choice") or "—"
    when = live.get("when") or datetime.now().strftime("%Y-%m-%d %H:%M")

    buf = io.BytesIO()
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    accent = "#0a5c56" if label != "Yes" else "#be123c"
    ax.add_patch(plt.Rectangle((0, 0.955), 1, 0.045, color=accent, transform=ax.transAxes, clip_on=False))
    ax.text(0.06, 0.972, "Heart Disease Risk  ·  Assessment summary", color="white",
            fontsize=13, fontweight="bold", va="center")
    ax.text(0.06, 0.925, f"{when}   ·   Model: {model}   ·   Prediction: {label}   ·   {pct:.0f}% risk",
            fontsize=10, color="#16232b")
    ax.text(0.06, 0.898, "Class project demo on 10,000 public records — not a medical diagnosis.",
            fontsize=8.5, color="#64748b")

    bar_x, bar_y, bar_w, bar_h = 0.06, 0.868, 0.72, 0.016
    n_seg = 40
    for i in range(n_seg):
        t = i / max(n_seg - 1, 1)
        if t < 0.5:
            u = t / 0.5
            color = (0.06 + 0.77 * u, 0.46 + 0.32 * u, 0.34 * (1 - u))
        else:
            u = (t - 0.5) / 0.5
            color = (0.83 + 0.05 * u, 0.78 * (1 - u), 0.34 * (1 - u))
        ax.add_patch(plt.Rectangle((bar_x + bar_w * i / n_seg, bar_y), bar_w / n_seg + 0.0005, bar_h,
                                   color=color, linewidth=0))
    marker_x = bar_x + bar_w * min(max(pct / 100.0, 0), 1)
    ax.plot([marker_x], [bar_y + bar_h + 0.006], marker="v", color="#16232b", markersize=7)
    ax.text(0.80, 0.874, f"{pct:.0f}%  ·  {label}", fontsize=9, fontweight="bold",
            color=accent, va="center")

    ax.text(0.06, 0.848, "What this means", fontsize=11, fontweight="bold", color="#16232b")
    y = 0.824
    for line in textwrap.wrap(str(reason.get("headline", "")).replace("**", ""), 92):
        ax.text(0.06, y, line, fontsize=9.5, color="#16232b")
        y -= 0.018
    for line in textwrap.wrap(str(reason.get("meaning", "")).replace("**", ""), 92):
        ax.text(0.06, y, line, fontsize=8.5, color="#475569")
        y -= 0.016

    y -= 0.012
    ax.text(0.06, y, "Patient inputs", fontsize=11, fontweight="bold", color="#16232b")
    y -= 0.01
    rows = []
    for col, val in raw.items():
        unit = FIELD_UNITS.get(col, "")
        shown = f"{val:g} {unit}".strip() if isinstance(val, (int, float, np.floating)) else str(val)
        rows.append((SHORT_LABELS.get(col, col), shown))
    col_x = (0.06, 0.54)
    row_h = 0.0165
    for i, (name, shown) in enumerate(rows):
        xx = col_x[0] if i < (len(rows) + 1) // 2 else col_x[1]
        yy = y - (i % ((len(rows) + 1) // 2)) * row_h
        ax.text(xx, yy, f"{name}:  {shown}", fontsize=8, color="#334155", va="top")
    y -= ((len(rows) + 1) // 2) * row_h + 0.02

    ax.text(0.06, y, "Why the model said that", fontsize=11, fontweight="bold", color="#16232b")
    y -= 0.018
    for line in textwrap.wrap(str(reason.get("why", "")).replace("**", ""), 92):
        ax.text(0.06, y, line, fontsize=8.5, color="#475569")
        y -= 0.016

    if history:
        y -= 0.012
        ax.text(0.06, y, "Visit history (this session)", fontsize=11, fontweight="bold", color="#16232b")
        y -= 0.008
        hist_h = 0.14
        axh = fig.add_axes([0.08, max(0.12, y - hist_h), 0.84, hist_h - 0.01])
        xs = [h["visit"] for h in history]
        ys = [h["risk"] for h in history]
        axh.plot(xs, ys, color=accent, marker="o", linewidth=2)
        axh.fill_between(xs, ys, color=accent, alpha=0.12)
        axh.set_ylim(0, 100)
        axh.set_xlabel("Visit")
        axh.set_ylabel("% risk")
        axh.grid(True, linestyle=":", alpha=0.4)
        y = max(0.12, y - hist_h) - 0.02

    ax.text(0.06, 0.055, str(reason.get("note", "")).replace("**", ""), fontsize=8, color="#64748b")
    ax.text(0.06, 0.032, "Visual risk: green/teal = lower estimate, crimson = higher estimate.",
            fontsize=8, color="#64748b")

    with PdfPages(buf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return buf.getvalue()


if page == "🏠 Home (Predict & Overview)":
    st.session_state.setdefault("risk_history", [])
    st.session_state.setdefault("patient_preset", CUSTOM_PRESET)
    if st.session_state.get("patient_preset") in (
        "Borderline Vitals",
        "Elevated Labs, Family History",
    ):
        st.session_state.patient_preset = "Adult With Some Risk Factors"
    _seed_form_defaults(raw_df, numeric_cols, categorical_cols)
    last_risk = st.session_state.get("last_risk_pct")
    last_label = st.session_state.get("last_risk_label")
    live = st.session_state.get("live_pred")

    st.title("Heart Disease Risk")
    n_home = len(raw_df)
    yes_home = int((raw_df[dp.TARGET_COL] == "Yes").sum())
    no_home = n_home - yes_home
    with st.container(key="quick_overview"):
        st.markdown("**📊 QUICK OVERVIEW**")
        qo_cards = [
            ("qo_patients", "blue", ":material/groups:", "Patients", f"{n_home:,}"),
            ("qo_no", "green", ":material/check_circle:", "No", f"{no_home / max(n_home, 1):.0%}"),
            ("qo_yes", "red", ":material/monitor_heart:", "Yes", f"{yes_home / max(n_home, 1):.0%}"),
            ("qo_models", "orange", ":material/hub:", "Models", "4"),
        ]
        for col, (key, color, icon, label, value) in zip(st.columns(4, gap="small"), qo_cards):
            with col:
                with st.container(border=True, key=key):
                    st.metric(f"{icon} {label}", f":{color}[**{value}**]")

    extra_num = [c for c in numeric_cols if c not in FEATURED_NUM]
    extra_cat = [c for c in categorical_cols if c not in FEATURED_CAT]

    with st.container(key="live_console"):
        st.markdown("### 💓 LIVE HEART RISK PREDICTOR")
        model_choice = st.selectbox(
            "🤖 Diagnostic Model",
            list(all_results.keys()),
            key="pred_model",
            persist_state="session",
        )
        st.selectbox(
            "👤 Patient profile",
            [CUSTOM_PRESET] + list(PATIENT_PRESETS.keys()),
            key="patient_preset",
            on_change=_on_patient_preset,
            persist_state="session",
            help="Loads every form field at once so a reviewer can try the model without typing 15+ values.",
        )
        best_model = all_results[model_choice]["best_model"]

        left, right = st.columns([1.2, 1.15], gap="large")
        with left:
            with st.container(key="live_vitals"):
                st.markdown("**💗 PATIENT VITALS**")
                v1, v2 = st.columns(2)
                with v1:
                    _num_widget("Age", raw_df["Age"].median())
                with v2:
                    bp_val = _num_widget("Blood Pressure", raw_df["Blood Pressure"].median())
                v3, v4 = st.columns(2)
                with v3:
                    chol_val = _num_widget("Cholesterol Level", raw_df["Cholesterol Level"].median())
                with v4:
                    bmi_val = _num_widget("BMI", raw_df["BMI"].median())
                v5, v6 = st.columns(2)
                with v5:
                    _num_widget("Sleep Hours", raw_df["Sleep Hours"].median())
                with v6:
                    if extra_num:
                        _num_widget(extra_num[0], raw_df[extra_num[0]].median())

            with st.container(key="live_history"):
                st.markdown("**🧬 PATIENT HISTORY & LIFESTYLE**")
                h_rows = [FEATURED_CAT[:2], FEATURED_CAT[2:4], FEATURED_CAT[4:6]]
                for row_cols in h_rows:
                    hcs = st.columns(2)
                    for i, col in enumerate(row_cols):
                        if col in dp.ORDINAL_MAPS:
                            options = list(dp.ORDINAL_MAPS[col].keys())
                        else:
                            options = sorted(raw_df[col].dropna().unique().tolist())
                        with hcs[i]:
                            _cat_widget(col, options)

            leftover_num = extra_num[1:]
            extra_items = [("num", col) for col in leftover_num]
            extra_cats = list(extra_cat)
            if leftover_num and "Sugar Consumption" in extra_cats:
                extra_cats.remove("Sugar Consumption")
                extra_items.append(("cat", "Sugar Consumption"))
            extra_items.extend(("cat", col) for col in extra_cats)
            if extra_items:
                with st.expander("More labs & lifestyle"):
                    for i in range(0, len(extra_items), 2):
                        pair = extra_items[i:i + 2]
                        ecs = st.columns(2)
                        for j, (kind, col) in enumerate(pair):
                            with ecs[j]:
                                if kind == "num":
                                    _num_widget(col, raw_df[col].median())
                                elif col in dp.ORDINAL_MAPS:
                                    _cat_widget(col, list(dp.ORDINAL_MAPS[col].keys()))
                                else:
                                    _cat_widget(
                                        col,
                                        sorted(raw_df[col].dropna().unique().tolist()),
                                    )

        with right:
            with st.container(key="live_stage"):
                st.caption("💓 3D / Animated HEART")
                last_tier = verdict_tier(last_risk) if last_risk is not None else None
                heart_3d.render_beating_heart(
                    risk_pct=last_risk,
                    label=last_label,
                    tier=last_tier["css"] if last_tier else None,
                    bpm=_estimated_bpm(),
                    key="home_heart_3d",
                    height=520,
                )
                with st.container(key="cta_row", horizontal=True, gap="small"):
                    analyze = st.button(
                        "Calculate Risk",
                        type="primary",
                        width="stretch",
                        key="analyze_heart",
                        help="Run the selected model on the current form.",
                    )
                    st.button(
                        "Reset Form",
                        width="stretch",
                        key="reset_form",
                        on_click=_reset_patient_form,
                        args=(raw_df, numeric_cols, categorical_cols),
                        help="Restore dataset defaults and clear the last assessment.",
                    )

                if not live:
                    with st.container(key="assess_ready"):
                        st.markdown("🫀 **Assessment Ready**")
                        st.caption("Fill the vitals, then press Calculate Risk.")
                else:
                    # Just a compact pointer here — the full result renders at
                    # full page width directly below the form (see below),
                    # instead of being squeezed into this half-width column,
                    # which left it tall and cramped while the left column
                    # went blank underneath.
                    tier = verdict_tier(live["prob_disease"] * 100)
                    with st.container(key=f"assess_{tier['css']}"):
                        st.markdown(f"{tier['icon']} **{tier['word']} — scored {live['prob_disease'] * 100:.0f}/100**")
                        st.caption("Full result and reasons are just below ↓")


                st.markdown("**Health Indicators**")
                bp_now = st.session_state.get("pred_num_Blood Pressure", bp_val)
                bmi_now = st.session_state.get("pred_num_BMI", bmi_val)
                chol_now = st.session_state.get("pred_num_Cholesterol Level", chol_val)
                bp_dot, bp_tag = _vital_flag("bp", bp_now)
                bmi_dot, bmi_tag = _vital_flag("bmi", bmi_now)
                chol_dot, chol_tag = _vital_flag("chol", chol_now)
                st.markdown(
                    f"- Blood Pressure: **{float(bp_now):.0f} mmHg** · {bp_dot} {bp_tag}  \n"
                    f"- Body Mass Index: **{float(bmi_now):.2f} kg/m²** · {bmi_dot} {bmi_tag}  \n"
                    f"- Cholesterol Level: **{float(chol_now):.0f} mg/dL** · {chol_dot} {chol_tag}"
                )
                history = st.session_state.get("risk_history") or []
                if history:
                    with st.container(border=True, key="visit_history"):
                        last_visit = history[-1]
                        st.markdown("**Visit history**")
                        st.caption(
                            f"{len(history)} visit{'s' if len(history) != 1 else ''} this session "
                            f"· latest {last_visit['risk']:.0f}% ({last_visit['label']})"
                        )
                        hist_df = pd.DataFrame(history)
                        st.line_chart(hist_df, x="visit", y="risk", height=96)

    if analyze:
        raw_input = {
            **{col: st.session_state[f"pred_num_{col}"] for col in numeric_cols},
            **{col: st.session_state[f"pred_cat_{col}"] for col in categorical_cols},
        }
        row = dp.build_single_row_features(raw_input, categorical_cols, X.columns.tolist())
        pred = int(best_model.predict(row)[0])
        prob_disease = float(best_model.predict_proba(row)[0, 1])
        label = le_target.inverse_transform([pred])[0]
        reason = explain_live_prediction(
            best_model, row, pred, prob_disease,
            raw_input=raw_input,
        )
        evidence_df, n_similar, n_rows = build_evidence_table(
            raw_input, raw_df, reason.get("highlight_fields"),
        )
        st.session_state.last_risk_pct = prob_disease * 100
        st.session_state.last_risk_label = label
        _record_visit(prob_disease, label, model_choice)
        st.session_state.live_pred = {
            "pred": pred,
            "prob_disease": prob_disease,
            "label": label,
            "model_choice": model_choice,
            "reason": reason,
            "evidence_df": evidence_df,
            "n_similar": n_similar,
            "n_rows": n_rows,
            "raw_input": raw_input,
            "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        st.rerun()

    if live:
        # Full page width here — not squeezed into the half-width right
        # column — but still directly below the form, not pushed down past
        # Health Indicators, Visit History, and Explore Further.
        pct = live["prob_disease"] * 100
        reason = live["reason"]
        evidence_df = live["evidence_df"]
        n_rows = live["n_rows"]
        tier = verdict_tier(pct)
        panel_key = f"explain_panel_{tier['css']}"

        st.subheader("Prediction Result")
        gauge_left, gauge_mid, gauge_right = st.columns([1, 2, 1])
        with gauge_mid:
            risk_gauge.render_total_risk_gauge(pct, live["label"], tier=tier["css"])

        with st.container(border=True, key=panel_key):
            st.markdown(f"### {tier['icon']} {tier['word']} — scored {pct:.0f}/100")
            st.caption(reason["meaning"])

            st.warning(
                "**How much to trust this:** testing across this whole project found the model "
                "barely beats a random guess at telling Yes from No. Treat this score as a "
                "talking point, not an answer.",
                icon="⚠️",
            )

            st.divider()

            col_bullets, col_table = st.columns([1, 1.3], gap="large")
            with col_bullets:
                st.markdown("**What pushed this score**")
                bullets = _reason_bullets(evidence_df)
                if bullets:
                    for b in bullets:
                        st.markdown(f"- {b}")
                else:
                    st.markdown(
                        "- None of your top answers looked very different from either group in "
                        "our records — that's a big part of *why* the model can't tell the two apart."
                    )
                st.caption(reason["why"])
            with col_table:
                st.markdown(f"**All {n_rows} answers compared to typical patients**")
                st.caption(
                    "Each row is one question from the form. "
                    "Compare **your answer** with typical people in our records who did / did not have heart disease."
                )
                st.dataframe(evidence_df, width="stretch", hide_index=True, height=280)

            st.divider()
            st.caption(reason["note"])

            pdf_bytes = build_assessment_pdf(live, st.session_state.get("risk_history") or [])
            st.download_button(
                "Export Assessment",
                data=pdf_bytes,
                file_name=f"heart_risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                width="stretch",
                icon=":material/download:",
                key="export_assessment_pdf",
                help="Download a printable PDF with inputs, model output, and the risk bar.",
            )

    st.divider()
    def _open_explore_page(page_name):
        st.session_state.current_page = page_name
        if page_name in MAIN_PAGES:
            st.session_state.main_page_radio = page_name
            st.session_state.more_page_radio = None
        else:
            st.session_state.more_page_radio = page_name

    with st.container(key="explore_further"):
        st.subheader("🧭 Explore Further")
        st.caption("Jump from this predictor into the analysis that sits behind it.")
        explore_col1, explore_col2 = st.columns(2, gap="medium")
        with explore_col1:
            with st.container(border=True, key="explore_eda"):
                st.badge("For reviewers", color="green")
                st.markdown("### 🔍 See the Dataset")
                st.caption(
                    "10,000 patient records, class balance, missing-value patterns, "
                    "correlations, and sample rows — the full exploratory analysis behind this predictor."
                )
                st.button(
                    "Open Exploratory Analysis",
                    key="go_eda",
                    width="stretch",
                    on_click=_open_explore_page,
                    args=("🔍 EDA",),
                )
        with explore_col2:
            with st.container(border=True, key="explore_models"):
                st.badge("For everyone", color="blue")
                st.markdown("### 📊 See the Models")
                st.caption(
                    "Full metrics for all 4 algorithms, and why Random Forest is the report's "
                    "recommended pick even though KNN scores a marginally higher ROC-AUC."
                )
                st.button(
                    "Open Model Comparison",
                    key="go_models",
                    width="stretch",
                    on_click=_open_explore_page,
                    args=("📊 Model Comparison",),
                )


elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.caption(
        "Raw data only — before imputation, encoding, or modelling. "
        "Use the snapshot, then open a category tab."
    )

    eda_figs = prefetch_eda_plots(raw_df, numeric_cols, categorical_cols)
    outlier_df, table_numeric, table_categorical, fig_assoc, _, _, _, _, _ = prefetch_stats(raw_df, numeric_cols, categorical_cols, X, y)

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
            st.markdown("**3D patient cloud**")
            st.caption(
                "Age × BMI × cholesterol for a sample of records. Rotate the cloud — "
                "if the two colors formed separate clusters, a classifier would have an easy job. "
                "Here they occupy almost the same space."
            )
            st.plotly_chart(get_patient_cloud_fig(raw_df, dp.TARGET_COL), width="stretch")

        with st.container(border=True):
            st.markdown("**Numeric features split by target**")
            st.caption("If a field predicted disease, the Yes and No boxes would sit at different levels. Overlapping boxes mean the two groups look the same.")
            st.pyplot(eda_figs["num_by_target"])

        with st.container(border=True):
            st.markdown("**Disease rate by category**")
            st.caption("Share of Yes within each category. If every bar is near the overall 20% rate, that category does not change risk.")
            st.pyplot(eda_figs["cat_rate"])

        with st.container(border=True):
            st.markdown("**Category counts by heart disease status**")
            st.caption("How many patients sit in each category, split into Yes and No. Taller No bars are expected because about 80% of the file is No.")
            st.pyplot(eda_figs["cat_counts"])

        with st.container(border=True):
            st.markdown("**Category mix (% Yes vs No)**")
            st.caption("Within each category, the share that is Yes vs No. The dashed line is the overall 20% Yes rate. Bars stuck near that line mean the category does not shift risk.")
            st.pyplot(eda_figs["cat_pct"])

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

    _, _, _, _, _, fig_mcar, fig_corr, anova_df, chi2_df = prefetch_stats(raw_df, numeric_cols, categorical_cols, X, y)

    with st.container(key="prep_category_tabs"):
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
        .st-key-eval_metrics_row [data-testid="stHorizontalBlock"] {
            align-items: stretch !important;
        }
        .st-key-eval_metrics_row [data-testid="stColumn"] {
            display: flex !important;
            flex-direction: column !important;
        }
        .st-key-eval_metrics_row [data-testid="stColumn"] > div,
        .st-key-compare_knn, .st-key-compare_logreg, .st-key-compare_rf, .st-key-compare_dt {
            height: 100% !important;
            flex: 1 1 auto !important;
        }
        .st-key-compare_knn, .st-key-compare_logreg, .st-key-compare_rf, .st-key-compare_dt {
            border-radius: 14px !important;
            box-shadow: 0 8px 18px rgba(22, 35, 43, 0.07) !important;
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
        .st-key-eval_metrics_row [data-testid="stMetric"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0.15rem 0.05rem 0.35rem !important;
            backdrop-filter: none !important;
        }
        .st-key-eval_metrics_row [data-testid="stMetric"]:hover {
            transform: none !important;
            box-shadow: none !important;
        }
        .st-key-eval_metrics_row [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
        }
        .st-key-eval_metrics_row [data-testid="stMetricValue"] {
            font-size: 1.55rem !important;
            font-weight: 800 !important;
        }
        .st-key-compare_knn [data-testid="stMetricValue"],
        .st-key-compare_knn [data-testid="stMetricValue"] p,
        .st-key-compare_knn [data-testid="stMetricValue"] .stMarkdownColoredText {
            color: #9f1239 !important;
        }
        .st-key-compare_logreg [data-testid="stMetricValue"],
        .st-key-compare_logreg [data-testid="stMetricValue"] p,
        .st-key-compare_logreg [data-testid="stMetricValue"] .stMarkdownColoredText {
            color: #0b5f4b !important;
        }
        .st-key-compare_rf [data-testid="stMetricValue"],
        .st-key-compare_rf [data-testid="stMetricValue"] p,
        .st-key-compare_rf [data-testid="stMetricValue"] .stMarkdownColoredText {
            color: #1e4e6b !important;
        }
        .st-key-compare_dt [data-testid="stMetricValue"],
        .st-key-compare_dt [data-testid="stMetricValue"] p,
        .st-key-compare_dt [data-testid="stMetricValue"] .stMarkdownColoredText {
            color: #8a5a00 !important;
        }
        .st-key-compare_knn_auc, .st-key-compare_logreg_auc,
        .st-key-compare_rf_auc, .st-key-compare_dt_auc {
            background: #f3f4f6 !important;
            border: 1px solid #e5e7eb !important;
            border-top: 1px solid #e5e7eb !important;
            border-radius: 12px !important;
            box-shadow: none !important;
            margin-top: 0.35rem !important;
        }
        .st-key-compare_knn_auc [data-testid="stMetric"],
        .st-key-compare_logreg_auc [data-testid="stMetric"],
        .st-key-compare_rf_auc [data-testid="stMetric"],
        .st-key-compare_dt_auc [data-testid="stMetric"] {
            padding: 0.15rem 0.2rem 0.1rem !important;
        }
        </style>
        """
    )
    with st.container(key="eval_metrics_row"):
        model_cols = st.columns(4, gap="medium")
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
                    for metric_name in ("Accuracy", "Precision", "Recall", "F1-Score"):
                        st.metric(metric_name, f":{accent}[**{row[metric_name]:.4f}**]")
                    with st.container(border=True, key=f"{theme['key']}_auc"):
                        st.metric("ROC-AUC", f":{accent}[**{row['ROC-AUC']:.4f}**]")

    st.dataframe(
        metrics_view.style
            .highlight_max(subset=best_metric_cols, color="#d4edda")
            .format({c: "{:.4f}" for c in best_metric_cols}),
        width="stretch",
        hide_index=True,
    )
    if recommended_row["Model"] != auc_leader_row["Model"]:
        st.success(
            f"🏆 **{recommended_row['Model']}** ({recommended_row['Pipeline']}) is the recommended model "
            f"(ROC-AUC **{recommended_row['ROC-AUC']:.4f}**). Although **{auc_leader_row['Model']}** has a slightly "
            f"higher ROC-AUC (**{auc_leader_row['ROC-AUC']:.4f}**), the difference is very small. "
            "Random Forest was chosen because it is easier to understand and handles imbalanced data better."
        )
    else:
        st.success(
            f"🏆 **{recommended_row['Model']}** ({recommended_row['Pipeline']}) is both the nominal ROC-AUC leader "
            f"and the report's recommended model (**{recommended_row['ROC-AUC']:.4f}**)."
        )

    st.subheader("📈 ROC-AUC of the 4 selected pipelines")
    auc_left, auc_mid, auc_right = st.columns([0.4, 3.2, 0.4])
    with auc_mid:
        st.pyplot(plot_selected_pipelines_roc_auc(
            basic_results,
            {
                "KNN": knn_basic_data["y_test"],
                "Logistic Regression": lr_basic_data["y_test"],
                "Random Forest": rf_basic_data["y_test"],
                "Decision Tree": dt_basic_data["y_test"],
            },
        ))

    st.subheader("🧭 Feature Importance (Top 10)")
    with st.container(border=True):
        with st.spinner("Computing feature importance for all 4 models..."):
            imp_knn = km.get_permutation_importance(best_results["KNN"]["best_model"], X_test, y_test).head(10)
            coef_lr = lgm.get_coefficients(best_results["Logistic Regression"]["best_model"], X.columns.tolist()).head(10)
            imp_rf = rfm.get_permutation_importance(best_results["Random Forest"]["best_model"], rf_X_test, rf_y_test).head(10)
            imp_dt = dtm.get_permutation_importance(best_results["Decision Tree"]["best_model"], dt_X_test, dt_y_test).head(10)

            feature_by_model = {
                "KNN": [f"{r.Feature} ({r.Importance:.3f})" for r in imp_knn.itertuples()],
                "Logistic Regression": [f"{r.Feature} ({r.Coefficient:+.3f})" for r in coef_lr.itertuples()],
                "Random Forest": [f"{r.Feature} ({r.Importance:.3f})" for r in imp_rf.itertuples()],
                "Decision Tree": [f"{r.Feature} ({r.Importance:.3f})" for r in imp_dt.itertuples()],
            }
            feature_summary = pd.DataFrame({
                "Rank": range(1, 11),
                **{name: feature_by_model[name] for name in MODEL_ORDER},
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

        st.subheader("📊 High-Level Metrics Impact : SMOTE vs. Basic")

        basic = results_df.iloc[0]
        smote = results_df.iloc[1]

        def get_delta(metric):
            return float(smote[metric] - basic[metric])

        render_smote_delta_kpis("knn", smote, get_delta)

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


        models_list = list(results.keys())
        res_basic = results[models_list[0]]
        res_smote = results[models_list[1]]
        render_basic_vs_smote_eval(
            "K-Nearest Neighbors",
            res_basic["metrics"],
            res_smote["metrics"],
            [
                ("Distance", "knn", "metric", "metric"),
                ("k (neighbors)", "knn", "n_neighbors", "plain"),
            ],
            "k and the distance metric",
        )


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


        lr_models_list = list(results_lr.keys())
        res_lr_basic = results_lr[lr_models_list[0]]
        res_lr_smote = results_lr[lr_models_list[1]]
        render_basic_vs_smote_eval(
            "Logistic Regression",
            res_lr_basic["metrics"],
            res_lr_smote["metrics"],
            [
                ("C", "logreg", "C", "plain"),
                ("Penalty", "logreg", "penalty", "penalty"),
                ("Solver", "logreg", "solver", "plain"),
                ("Class Weight", "logreg", "class_weight", "weight"),
            ],
            "C, penalty, solver, and class weight",
        )

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


        rf_models_list = list(rf_results.keys())
        rf_res_basic = rf_results[rf_models_list[0]]
        rf_res_smote = rf_results[rf_models_list[1]]
        render_basic_vs_smote_eval(
            "Random Forest",
            rf_res_basic["metrics"],
            rf_res_smote["metrics"],
            [
                ("Trees", "rf", "n_estimators", "plain"),
                ("Max Depth", "rf", "max_depth", "plain"),
                ("Min Samples Leaf", "rf", "min_samples_leaf", "plain"),
                ("Class Weight", "rf", "class_weight", "weight"),
            ],
            "number of trees, max depth, min samples leaf, and class weight",
        )


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


        dt_models_list = list(dt_results.keys())
        dt_res_basic = dt_results[dt_models_list[0]]
        dt_res_smote = dt_results[dt_models_list[1]]
        render_basic_vs_smote_eval(
            "Decision Tree",
            dt_res_basic["metrics"],
            dt_res_smote["metrics"],
            [
                ("Criterion", "dt", "criterion", "plain"),
                ("Max Depth", "dt", "max_depth", "plain"),
                ("Min Samples Leaf", "dt", "min_samples_leaf", "plain"),
            ],
            "criterion, max depth, and min samples leaf",
        )


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