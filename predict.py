
import sys
import mlflow
import mlflow.sklearn
import pandas as pd

FEATURES = [
    "amount",
    "hour",
    "merchant_risk",
    "account_age_days",
]


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py YOUR_RUN_ID")
        print("Use the Run ID printed by train.py.")
        return

    run_id = sys.argv[1]
    model_uri = f"runs:/{run_id}/fraud_model"

    print("Loading trained model from MLflow...")

    model = mlflow.sklearn.load_model(model_uri)

    # Example transaction; change values to test other cases.
    transaction = pd.DataFrame([{
        "amount": 2500.00,
        "hour": 2,
        "merchant_risk": 0.85,
        "account_age_days": 10,
    }], columns=FEATURES)

    prediction = int(model.predict(transaction)[0])
    probabilities = model.predict_proba(transaction)[0]

    fraud_probability = float(probabilities[1])

    print("\nBankShield Fraud Detection Result")
    print("-" * 40)
    print(f"Transaction amount: {transaction.iloc[0]['amount']}")
    print(f"Transaction hour: {transaction.iloc[0]['hour']}")
    print(f"Fraud probability: {fraud_probability:.2%}")
    print(
        "Prediction:",
        "POTENTIAL FRAUD" if prediction == 1 else "LIKELY LEGITIMATE"
    )


if __name__ == "__main__":
    main()
