"""
Classification pipeline: Logistic Regression + Random Forest
comparing CA / PD / NH groups on non-motor features.

Usage:
    python classification_pipeline.py

Data files are expected in DATA_DIR (see CONFIG section below).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # <-- was missing, RF section would have crashed
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
    balanced_accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# =========================
# CONFIG - portable paths
# =========================
# BASE_DIR = folder this script lives in
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

N_BOOT = 1000        # bootstrap resamples for CI estimation
RANDOM_STATE = 42

# Single source file
DATA_FILE = DATA_DIR / "Data.xlsx"
SHEET_NAME = "Data"
GROUP_COL = "Group"


# =========================
# Define all classification tasks
# =========================

comparisons = {
    'CA vs NH': {
        'filter': lambda df: df[df[GROUP_COL] != 'PD'],
        'map': {'control': 0, 'CA': 1}
    },
    'PD vs NH': {
        'filter': lambda df: df[df[GROUP_COL] != 'CA'],
        'map': {'control': 0, 'PD': 1}
    },
    'PD vs CA': {
        'filter': lambda df: df[df[GROUP_COL] != 'control'],
        'map': {'CA': 0, 'PD': 1}
    }
}

COLORS = {
    'CA vs NH': "#0072B2",
    'PD vs NH': '#FC8D62',
    'PD vs CA': '#66C2A5'
}


def load_full_data():
    """Load Data.xlsx once."""
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_and_clean(cfg, full_df):
    """Apply one comparison's group filter + label map to the shared dataframe,
    keep only feature columns, drop NaNs."""
    df = cfg['filter'](full_df) if cfg['filter'] else full_df

    y = df[GROUP_COL].map(cfg['map'])
    X = df.drop(columns=[GROUP_COL], errors='ignore')
    X = X.apply(pd.to_numeric, errors='coerce')

    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=[GROUP_COL])
    y = data[GROUP_COL]
    return X, y


def bootstrap_oob_ci(pipeline_or_model, X, y, n_boot=N_BOOT, random_state=RANDOM_STATE):
    """
    Bootstrap resample with replacement, evaluate on out-of-bag (OOB) samples.
    Returns arrays of per-resample accuracy and AUC (used for CIs and SDs).
    """
    rng = np.random.RandomState(random_state)
    n = len(y)
    boot_acc, boot_auc = [], []

    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        X_boot, y_boot = X.iloc[idx], y.iloc[idx]

        oob_mask = np.ones(n, dtype=bool)
        oob_mask[idx] = False
        if np.sum(oob_mask) == 0:
            continue

        X_oob, y_oob = X.iloc[oob_mask], y.iloc[oob_mask]

        # Skip resamples that end up single-class (can't compute AUC / fit properly)
        if len(np.unique(y_boot)) < 2 or len(np.unique(y_oob)) < 2:
            continue

        pipeline_or_model.fit(X_boot, y_boot)
        y_pred_boot = pipeline_or_model.predict(X_oob)
        y_proba_boot = pipeline_or_model.predict_proba(X_oob)[:, 1]

        boot_acc.append(np.mean(y_pred_boot == y_oob))
        boot_auc.append(roc_auc_score(y_oob, y_proba_boot))

    return np.array(boot_acc), np.array(boot_auc)


def run_logistic_regression(full_df):
    results = []
    roc_data = {}
    perm_importance_data = {}

    for name, cfg in comparisons.items():
        X, y = load_and_clean(cfg, full_df)
        feature_names = X.columns

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(penalty='l2', solver='liblinear', random_state=RANDOM_STATE))
        ])
        loo = LeaveOneOut()

        # LOOCV performance metrics
        cv_scores = cross_val_score(pipeline, X, y, cv=loo, scoring='accuracy')
        y_pred = cross_val_predict(pipeline, X, y, cv=loo)
        y_proba = cross_val_predict(pipeline, X, y, cv=loo, method='predict_proba')[:, 1]

        acc = np.mean(cv_scores)
        acc_sd = np.std(cv_scores, ddof=1)
        acc_sem = acc_sd / np.sqrt(len(cv_scores))
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_data[name] = (fpr, tpr, auc)

        # Bootstrap CIs (separate OOB evaluation, not the same as LOOCV above)
        boot_acc, boot_auc = bootstrap_oob_ci(pipeline, X, y)
        acc_ci_low, acc_ci_high = np.percentile(boot_acc, [2.5, 97.5])
        auc_ci_low, auc_ci_high = np.percentile(boot_auc, [2.5, 97.5])
        acc_boot_sd = np.std(boot_acc, ddof=1)
        auc_boot_sd = np.std(boot_auc, ddof=1)
        print(f"{name}: Boot samples used = {len(boot_acc)}")

        results.append({
            'Comparison': name, 'N': len(y), 'Accuracy': acc, 'Accuracy_SD': acc_sd,
            'Accuracy_SEM': acc_sem, 'Balanced_Accuracy': bal_acc, 'Precision': prec,
            'Recall': rec, 'F1': f1, 'AUC': auc, 'boot_auc': np.mean(boot_auc),
            'boot_acc': np.mean(boot_acc), 'Accuracy_Boot_SD': acc_boot_sd,
            'Accuracy_CI_low': acc_ci_low, 'Accuracy_CI_high': acc_ci_high,
            'AUC_Boot_SD': auc_boot_sd, 'AUC_CI_low': auc_ci_low, 'AUC_CI_high': auc_ci_high,
        })

        # Fit on full data (interpretation only - not for performance claims)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        final_model = LogisticRegression(penalty='l2', solver='liblinear', random_state=RANDOM_STATE)
        final_model.fit(X_scaled, y)

        perm = permutation_importance(
            final_model, X_scaled, y, scoring='roc_auc', n_repeats=100, random_state=RANDOM_STATE
        )
        perm_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance_Mean': perm.importances_mean,
            'Importance_SD': perm.importances_std
        }).sort_values(by='Importance_Mean', ascending=False)
        perm_importance_data[name] = perm_df

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'classification_summary_LR.csv', index=False)

    plot_roc(roc_data, 'Logistic Regression ROC Curves', 'ROC_Curves_LR.png')
    plot_permutation_importance(perm_importance_data, 'Permutation_Importance_LR.png')


def run_random_forest(full_df):
    results = []
    roc_data = {}
    perm_importance_data = {}

    for name, cfg in comparisons.items():
        X, y = load_and_clean(cfg, full_df)

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))
        ])
        loo = LeaveOneOut()

        cv_scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
        y_proba = cross_val_predict(model, X, y, cv=loo, method='predict_proba')[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        acc = np.mean(cv_scores)
        auc = roc_auc_score(y, y_proba)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_data[name] = (fpr, tpr, auc)

        boot_acc, boot_auc = bootstrap_oob_ci(model, X, y)
        acc_ci_low, acc_ci_high = np.percentile(boot_acc, [2.5, 97.5])
        auc_ci_low, auc_ci_high = np.percentile(boot_auc, [2.5, 97.5])
        auc_boot_sd = np.std(boot_auc, ddof=1)
        print(f"{name}: Boot samples used = {len(boot_acc)}")

        results.append({
            'Comparison': name, 'Accuracy': acc, 'Accuracy_CI_low': acc_ci_low,
            'Accuracy_CI_high': acc_ci_high, 'AUC': auc, 'AUC_Boot_SD': auc_boot_sd,
            'AUC_CI_low': auc_ci_low, 'AUC_CI_high': auc_ci_high,
            'Precision': precision, 'Recall': recall
        })

        # Fit on full data for permutation importance (scoring='roc_auc' - kept consistent with LR)
        model.fit(X, y)
        perm = permutation_importance(
            model, X, y, scoring='roc_auc', n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
        )
        perm_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance_Mean': perm.importances_mean,
            'Importance_SD': perm.importances_std
        }).sort_values(by='Importance_Mean', ascending=False)
        perm_importance_data[name] = perm_df

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'classification_summary_RF.csv', index=False)

    plot_roc(roc_data, 'Random Forest ROC Curves', 'ROC_Curves_RF.png')
    plot_permutation_importance(perm_importance_data, 'Permutation_Importance_RF.png')


def plot_roc(roc_data, title, filename):
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=COLORS[name], linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.show()


def plot_permutation_importance(perm_importance_data, filename):
    all_features = sorted(set().union(*[df['Feature'] for df in perm_importance_data.values()]))
    importance_df = pd.DataFrame(index=all_features)
    for name, perm_df in perm_importance_data.items():
        importance_df[name] = perm_df.set_index('Feature')['Importance_Mean']
    importance_df = importance_df.fillna(0)
    importance_df.to_csv(OUTPUT_DIR / filename.replace('.png', '.csv'))

    importance_df.plot(kind='barh', figsize=(11, 9), color=list(COLORS.values()))
    plt.title('Permutation Feature Importances')
    plt.xlabel('Mean Decrease in AUC')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    full_df = load_full_data()
    run_logistic_regression(full_df)
    run_random_forest(full_df)
