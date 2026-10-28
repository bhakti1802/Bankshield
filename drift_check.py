
import os
import numpy as np
import pandas as pd
import mlflow

from scipy.stats import ks_2samp

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

REFERENCE_FILE = os.path.join(DATA_DIR, "reference.csv")
CURRENT_FILE = os.path.join(DATA_DIR, "current.csv")
REPORT_FILE = os.path.join(DATA_DIR, "drift_report.csv")

FEATURES = [
    "amount",
    "hour",
    "merchant_risk",
    "account_age_days",
]

P_VALUE_THRESHOLD = 0.05


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

def load_data():
    if not os.path.exists(REFERENCE_FILE):
        raise FileNotFoundError(
            f"Reference file not found: {REFERENCE_FILE}. "
            "Run python train.py first."
        )

    if not os.path.exists(CURRENT_FILE):
        raise FileNotFoundError(
            f"Current file not found: {CURRENT_FILE}. "
            "Run python train.py first."
        )

    reference = pd.read_csv(REFERENCE_FILE)
    current = pd.read_csv(CURRENT_FILE)

    for feature in FEATURES:
        if feature not in reference.columns:
            raise ValueError(
                f"Feature '{feature}' missing from reference.csv"
            )

        if feature not in current.columns:
            raise ValueError(
                f"Feature '{feature}' missing from current.csv"
            )

    return reference, current


# --------------------------------------------------
# 3. Detect data drift
# --------------------------------------------------

def detect_drift(reference, current):
    results = []

    for feature in FEATURES:
        reference_values = (
            pd.to_numeric(reference[feature], errors="coerce")
            .dropna()
            .to_numpy()
        )

        current_values = (
            pd.to_numeric(current[feature], errors="coerce")
            .dropna()
            .to_numpy()
        )

        if len(reference_values) == 0 or len(current_values) == 0:
            results.append({
                "feature": feature,
                "reference_count": len(reference_values),
                "current_count": len(current_values),
                "ks_statistic": np.nan,
                "p_value": np.nan,
                "drift_detected": "UNKNOWN",
            })
            continue

        statistic, p_value = ks_2samp(
            reference_values,
            current_values,
        )

        drift_detected = p_value < P_VALUE_THRESHOLD

        results.append({
            "feature": feature,
            "reference_count": len(reference_values),
            "current_count": len(current_values),
            "ks_statistic": round(float(statistic), 6),
            "p_value": float(p_value),
            "drift_detected": (
                "YES" if drift_detected else "NO"
            ),
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# 4. Save report and log results to MLflow
# --------------------------------------------------

def main():
    print("=" * 55)
    print("BANKSHIELD - DATA DRIFT DETECTION")
    print("=" * 55)

    print("\nLoading reference and current datasets...")

    reference, current = load_data()

    print(f"Reference records: {len(reference)}")
    print(f"Current records:   {len(current)}")

    print("\nRunning Kolmogorov-Smirnov tests...")

    report = detect_drift(reference, current)

    report.to_csv(REPORT_FILE, index=False)

    mlflow.set_experiment("BankShield_Fraud_Drift")

    with mlflow.start_run(
        run_name="Data_Drift_Detection"
    ):
        mlflow.log_param(
            "drift_test",
            "Kolmogorov-Smirnov",
        )

        mlflow.log_param(
            "p_value_threshold",
            P_VALUE_THRESHOLD,
        )

        total_tested = int(
            report["p_value"].notna().sum()
        )

        drifted_features = int(
            (report["drift_detected"] == "YES").sum()
        )

        mlflow.log_metric(
            "features_tested",
            total_tested,
        )

        mlflow.log_metric(
            "features_with_drift",
            drifted_features,
        )

        for _, row in report.iterrows():
            feature = row["feature"]

            if pd.notna(row["ks_statistic"]):
                mlflow.log_metric(
                    f"ks_statistic_{feature}",
                    float(row["ks_statistic"]),
                )

            if pd.notna(row["p_value"]):
                mlflow.log_metric(
                    f"p_value_{feature}",
                    float(row["p_value"]),
                )

        mlflow.log_artifact(REPORT_FILE)

    print("\nDrift detection results")
    print("-" * 70)
    print(report.to_string(index=False))

    print(f"\nReport saved to: {REPORT_FILE}")
    print(f"Features tested: {total_tested}")
    print(f"Features with detected drift: {drifted_features}")
    print("\nDrift detection completed!")


if __name__ == "__main__":
    main()
