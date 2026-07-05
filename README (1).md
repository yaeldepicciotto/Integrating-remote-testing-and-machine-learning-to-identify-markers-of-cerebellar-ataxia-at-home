# Classification Pipeline (Logistic Regression + Random Forest)

Code for the group classification analyses (CA vs NH, PD vs NH, PD vs CA) reported in [paper name/citation].

## Data

`data/Data.xlsx` contains the data used in this analysis, corresponding to the Supplementary Material of the paper
(see [link/DOI once available]).

Columns `Age`, `Gender`, `YoE`, `Motor`, `Disease Duration`, and `total_score_ISEL` are excluded from the model
features (see `NON_FEATURE_COLS` in `classification_pipeline_digital_markers.py`); the remaining columns
(MoCA, Depression_tscore, Anxiety_tscore, personality trait scores, ISEL sub-scores) are used as predictors,
with `Group` (control / CA / PD) as the classification target.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python classification_pipeline_digital_markers.py
```

This runs both the Logistic Regression and Random Forest analyses for all three group comparisons, and writes
to an `outputs/` folder (auto-created):
- `classification_summary_LR.csv`, `classification_summary_RF.csv` — performance metrics (accuracy, AUC,
  precision, recall, bootstrap CIs)
- `ROC_Curves_LR.png`, `ROC_Curves_RF.png`
- `Permutation_Importance_LR.png`, `Permutation_Importance_RF.png` (+ underlying CSVs)

## Notes on method

- Performance metrics (accuracy, AUC, etc.) are estimated via Leave-One-Out Cross-Validation (LOOCV), appropriate
  given small sample sizes.
- 95% confidence intervals are estimated separately via bootstrap resampling (1000 resamples) with out-of-bag
  evaluation.
- Permutation feature importance is computed with `scoring='roc_auc'` for both models, so importances are
  comparable across Logistic Regression and Random Forest.
