# Classification Pipeline (Logistic Regression + Random Forest)

Code for the group classification analyses (CA vs NH, PD vs NH, PD vs CA) reported in "Integrating-remote-testing-and-machine-learning-to-identify-markers-of-cerebellar-ataxia-at-home".

## Data

`data/Data.xlsx` contains the data used in this analysis, corresponding to the Supplementary Material of the paper.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python classification_pipeline_digital_markers.py
```

This runs both the Logistic Regression and Random Forest analyses for all three group comparisons.
