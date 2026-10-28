
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FEATURES = [
    "amount",
    "hour",
    "merchant_risk",
    "account_age_days",
]


# --------------------------------------------------
# 2. Generate synthetic transaction data
# --------------------------------------------------

def generate_transactions(n=2000, seed=42, drift=False):
    rng = np.random.default_rng(seed)

    if not drift:
        # Normal transaction distribution
        amount = rng.lognormal(
            mean=3.5,
            sigma=1.0,
            size=n,
        )

        hour_probs = np.array([
            0.10, 0.08, 0.07, 0.06, 0.05, 0.04,
            0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
            0.03, 0.03, 0.03, 0.03, 0.04, 0.04,
            0.04, 0.04, 0.05, 0.06, 0.07, 0.08
        ])

        # Normalize probabilities to sum exactly to 1
        hour_probs = hour_probs / hour_probs.sum()

        hour = rng.choice(
            np.arange(24),
            size=n,
            p=hour_probs,
        )

        merchant_risk = rng.beta(3, 2, size=n)

        account_age_days = rng.integers(
            1,
            3650,
            size=n,
        )

    else:
        # Shifted distribution to simulate data drift
        amount = rng.lognormal(
            mean=4.5,
            sigma=0.9,
            size=n,
        )

        hour = rng.integers(0, 24, size=n)

        merchant_risk = rng.beta(2, 5, size=n)

        account_age_days = rng.integers(
            1,
            4000,
            size=n,
        )

    # Simulated fraud probability
    logit = (
        -3.2
        + 0.0009 * amount
        + 2.0 * merchant_risk
        + 0.8 * (hour <= 4).astype(float)
        + 0.8 * (account_age_days < 30).astype(float)
    )

    probability = 1 / (1 + np.exp(-logit))

    is_fraud = rng.binomial(
        1,
        probability,
    )

    return pd.DataFrame({
        "amount": amount.round(2),
        "hour": hour,
        "merchant_risk": merchant_risk.round(4),
        "account_age_days": account_age_days,
        "is_fraud": is_fraud,
    })


# --------------------------------------------------
# 3. Train model and track experiment with MLflow
# --------------------------------------------------

def main():
    print("=" * 55)
    print("BANKSHIELD - FRAUD DETECTION MLOPS")
    print("=" * 55)

    # Generate datasets
    print("\n1. Generating synthetic transaction data...")

    data = generate_transactions(
        n=2000,
        seed=42,
        drift=False,
    )

    current = generate_transactions(
        n=1000,
        seed=99,
        drift=True,
    )

    # Save the full and drifted datasets
    data.to_csv(
        os.path.join(DATA_DIR, "transactions.csv"),
        index=False,
    )

    current.to_csv(
        os.path.join(DATA_DIR, "current.csv"),
        index=False,
    )

    # Split features and target
    X = data[FEATURES]
    y = data["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Save training and testing datasets
    train_df = X_train.copy()
    train_df["is_fraud"] = y_train

    test_df = X_test.copy()
    test_df["is_fraud"] = y_test

    train_df.to_csv(
        os.path.join(DATA_DIR, "train.csv"),
        index=False,
    )

    test_df.to_csv(
        os.path.join(DATA_DIR, "test.csv"),
        index=False,
    )

    # Reference dataset contains feature columns only
    X_train.to_csv(
        os.path.join(DATA_DIR, "reference.csv"),
        index=False,
    )

    print(f"Total transactions : {len(data)}")
    print(f"Training records   : {len(X_train)}")
    print(f"Testing records    : {len(X_test)}")
    print(f"Training fraud rate: {y_train.mean():.3f}")

    # Configure MLflow experiment
    mlflow.set_experiment("BankShield_Fraud_Drift")

    # Start an MLflow run
    with mlflow.start_run(
        run_name="RandomForest_Fraud_Classifier"
    ) as run:

        print("\n2. Training Random Forest model...")

        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        # Model predictions
        predictions = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        # Log parameters to MLflow
        mlflow.log_param(
            "model_type",
            "RandomForestClassifier",
        )

        mlflow.log_param(
            "n_estimators",
            150,
        )

        mlflow.log_param(
            "max_depth",
            10,
        )

        mlflow.log_param(
            "test_size",
            0.20,
        )

        mlflow.log_param(
            "dataset_type",
            "synthetic",
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES),
        )

        # Log evaluation metrics to MLflow
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

        # Generate classification report
        report = classification_report(
            y_test,
            predictions,
            zero_division=0,
        )

        report_path = os.path.join(
            DATA_DIR,
            "classification_report.txt",
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(report)

        # Generate confusion matrix
        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        cm_path = os.path.join(
            DATA_DIR,
            "confusion_matrix.txt",
        )

        with open(
            cm_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                "Confusion Matrix\n"
                "Labels: 0 = Legitimate, 1 = Fraud\n\n"
            )
            file.write(str(cm))

        # Log evaluation artifacts
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)

        # Log trained model to MLflow
        mlflow.sklearn.log_model(
           sk_model=model,
           artifact_path="fraud_model",
           serialization_format="cloudpickle",
       )

        # Display evaluation results
        print("\n3. Model evaluation")
        print("-" * 35)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        print("\nClassification Report:")
        print(report)

        print("\nConfusion Matrix:")
        print(cm)

        print("\n4. MLflow tracking")
        print("-" * 35)
        print("Experiment: BankShield_Fraud_Drift")
        print("Run ID:", run.info.run_id)
        print("Model artifact: fraud_model")

        print("\n5. Generated CSV files")
        print("-" * 35)
        print("data/transactions.csv")
        print("data/train.csv")
        print("data/test.csv")
        print("data/reference.csv")
        print("data/current.csv")

        print("\nTraining completed successfully!")


# --------------------------------------------------
# 4. Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()
