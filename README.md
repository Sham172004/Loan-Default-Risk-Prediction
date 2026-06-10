# Loan Default Prediction for Credit Risk Assessment

**Domain:** Banking / Credit Risk  
**Institution:** XYZ Bank (mock project)

## Business Problem

XYZ Bank faces rising loan defaults (~18% observed), revenue loss from bad loans, and a rule-based approval system that misses complex risk patterns. This project builds a data-driven default prediction model to support smarter loan approval decisions.

## Dataset

| File | Records | Features | Target |
|------|---------|----------|--------|
| `Loan_default.csv` | 255,347 | 17 (+ engineered) | `Default` (0/1) |

**Default rate:** ~11.6% (imbalanced classification)

## Project Structure

```
Loan Default Risk Prediction/
├── main.ipynb          # Full analysis & modeling pipeline
├── Loan_default.csv    # Raw dataset
├── requirements.txt    # Python dependencies
├── predict.py          # CLI to score new loan applications
├── models/             # Saved artifacts (after running notebook)
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── model_config.pkl
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```

## Methodology

1. **EDA** — Class imbalance, credit score, DTI, income, employment, and categorical risk drivers
2. **Preprocessing** — Per-column label encoding, feature engineering, train/val/test split (70/15/15)
3. **Scaling & SMOTE** — StandardScaler on numeric features; SMOTE on training set only
4. **Models** — Logistic Regression, Random Forest, XGBoost
5. **Evaluation** — Accuracy, Precision, Recall, ROC-AUC, confusion matrix
6. **Threshold tuning** — F1-optimal decision threshold on validation set
7. **Risk segmentation** — Low / Medium / High risk buckets from predicted probability
8. **Business recommendations** — Actionable policy guidance for the Chief Risk Officer

## Key Deliverables

| Deliverable | Location in Notebook |
|-------------|---------------------|
| Predictive model | XGBoost (best ROC-AUC) saved to `models/` |
| Feature importance | RF & XGBoost importance plots |
| Risk categorization | Low / Medium / High segments with validation |
| Threshold recommendation | Data-driven F1-optimal threshold |
| Business recommendations | Final section |

## Models Compared

All three classifiers required by the project brief are trained and compared on the same validation split using identical metrics.

## Score a New Application

After running the notebook (or using pre-saved models in `models/`):

```bash
python predict.py --age 46 --income 84208 --loan-amount 129188 --credit-score 451 \
  --months-employed 26 --num-credit-lines 3 --interest-rate 21.17 --loan-term 24 \
  --dti-ratio 0.31 --employment-type Unemployed --education "Master's" \
  --marital-status Divorced --has-mortgage Yes --has-dependents Yes \
  --loan-purpose Auto --has-cosigner No
```
