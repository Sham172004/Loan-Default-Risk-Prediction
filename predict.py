"""
Score a new loan application using the trained XGBoost model.

Usage (terminal):
    python predict.py --age 46 --income 84208 --loan-amount 129188 --credit-score 451 \\
        --months-employed 26 --num-credit-lines 3 --interest-rate 21.17 --loan-term 24 \\
        --dti-ratio 0.31 --employment-type Unemployed --education "Master's" \\
        --marital-status Divorced --has-mortgage Yes --has-dependents Yes \\
        --loan-purpose Auto --has-cosigner No

Quick demo (no arguments needed):
    python predict.py --demo
"""
import argparse
import sys
import joblib
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
REQUIRED_MODEL_FILES = [
    "xgb_model.pkl",
    "scaler.pkl",
    "label_encoders.pkl",
    "model_config.pkl",
]

DEMO_ARGS = dict(
    age=46,
    income=84208,
    loan_amount=129188,
    credit_score=451,
    months_employed=26,
    num_credit_lines=3,
    interest_rate=21.17,
    loan_term=24,
    dti_ratio=0.31,
    education="Master's",
    employment_type="Unemployed",
    marital_status="Divorced",
    has_mortgage="Yes",
    has_dependents="Yes",
    loan_purpose="Auto",
    has_cosigner="No",
)


def check_models_exist():
    missing = [f for f in REQUIRED_MODEL_FILES if not (MODELS_DIR / f).exists()]
    if missing:
        print("ERROR: Model files not found.\n")
        print(f"Expected folder: {MODELS_DIR.resolve()}\n")
        print("Missing files:")
        for f in missing:
            print(f"  - {f}")
        print(
            "\nFix: Open main.ipynb and run all cells (Kernel -> Restart & Run All).\n"
            "     That saves the trained model into the models/ folder."
        )
        sys.exit(1)


def load_artifacts():
    check_models_exist()
    model = joblib.load(MODELS_DIR / "xgb_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    encoders = joblib.load(MODELS_DIR / "label_encoders.pkl")
    config = joblib.load(MODELS_DIR / "model_config.pkl")
    return model, scaler, encoders, config


def risk_segment(prob, low_cutoff, high_cutoff=0.70):
    if prob < low_cutoff:
        return "Low Risk"
    if prob < high_cutoff:
        return "Medium Risk"
    return "High Risk"


def build_features(args, encoders):
    row = {
        "Age": args.age,
        "Income": args.income,
        "LoanAmount": args.loan_amount,
        "CreditScore": args.credit_score,
        "MonthsEmployed": args.months_employed,
        "NumCreditLines": args.num_credit_lines,
        "InterestRate": args.interest_rate,
        "LoanTerm": args.loan_term,
        "DTIRatio": args.dti_ratio,
        "Education": args.education,
        "EmploymentType": args.employment_type,
        "MaritalStatus": args.marital_status,
        "HasMortgage": args.has_mortgage,
        "HasDependents": args.has_dependents,
        "LoanPurpose": args.loan_purpose,
        "HasCoSigner": args.has_cosigner,
    }
    df = pd.DataFrame([row])

    df["LoanToIncomeRatio"] = df["LoanAmount"] / df["Income"]
    df["MonthlyPaymentEst"] = (df["LoanAmount"] / df["LoanTerm"]) * (1 + df["InterestRate"] / 100)
    df["CreditRiskFlag"] = (df["CreditScore"] < 600).astype(int)
    df["HighDTIFlag"] = (df["DTIRatio"] > 0.5).astype(int)
    df["UnemployedFlag"] = (df["EmploymentType"] == "Unemployed").astype(int)
    df["ShortEmploymentFlag"] = (df["MonthsEmployed"] < 12).astype(int)

    for col, le in encoders.items():
        if row[col] not in le.classes_:
            raise ValueError(
                f"Invalid value '{row[col]}' for '{col}'.\n"
                f"Allowed values: {list(le.classes_)}"
            )
        df[col] = le.transform([row[col]])

    return df


def score(args):
    model, scaler, encoders, config = load_artifacts()
    df = build_features(args, encoders)
    df[config["numerical_cols"]] = scaler.transform(df[config["numerical_cols"]])

    prob = model.predict_proba(df[config["feature_columns"]])[:, 1][0]
    threshold = config["best_threshold"]
    decision = "APPROVE" if prob < threshold else "REVIEW/DECLINE"
    segment = risk_segment(prob, threshold, config.get("risk_high_cutoff", 0.70))

    print(f"Default Probability: {prob:.2%}")
    print(f"Risk Category:       {segment}")
    print(f"Decision (threshold={threshold:.2f}): {decision}")


def main():
    # Clicking "Run" in the IDE runs with no CLI args — use demo mode
    if len(sys.argv) == 1:
        print("No arguments provided. Running demo with a sample high-risk applicant.\n")
        print("Tip: use --help to see all options, or pass your own --age, --income, etc.\n")
        score(argparse.Namespace(**DEMO_ARGS))
        return

    parser = argparse.ArgumentParser(
        description="Predict loan default risk for a new application",
        epilog="Example: python predict.py --demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with a built-in sample applicant (no other args needed)",
    )
    parser.add_argument("--age", type=int)
    parser.add_argument("--income", type=float)
    parser.add_argument("--loan-amount", type=float)
    parser.add_argument("--credit-score", type=int)
    parser.add_argument("--months-employed", type=int)
    parser.add_argument("--num-credit-lines", type=int)
    parser.add_argument("--interest-rate", type=float)
    parser.add_argument("--loan-term", type=int)
    parser.add_argument("--dti-ratio", type=float)
    parser.add_argument("--education", default="Bachelor's")
    parser.add_argument("--employment-type", default="Full-time")
    parser.add_argument("--marital-status", default="Married")
    parser.add_argument("--has-mortgage", choices=["Yes", "No"], default="No")
    parser.add_argument("--has-dependents", choices=["Yes", "No"], default="No")
    parser.add_argument("--loan-purpose", default="Other")
    parser.add_argument("--has-cosigner", choices=["Yes", "No"], default="No")
    args = parser.parse_args()

    if args.demo:
        print("Demo mode — sample high-risk applicant:\n")
        score(argparse.Namespace(**DEMO_ARGS))
        return

    required = [
        "age", "income", "loan_amount", "credit_score", "months_employed",
        "num_credit_lines", "interest_rate", "loan_term", "dti_ratio",
    ]
    missing = [f"--{r.replace('_', '-')}" for r in required if getattr(args, r) is None]
    if missing:
        print("ERROR: Missing required arguments.\n")
        print("Required:", ", ".join(missing))
        print("\nQuick fix — run demo mode:")
        print("  python predict.py --demo")
        print("\nOr pass all required fields. See:")
        print("  python predict.py --help")
        sys.exit(1)

    score(args)


if __name__ == "__main__":
    main()
