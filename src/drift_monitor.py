
from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
DB_PATH = ROOT / "data" / "predictions.db"
REPORT_PATH = ROOT / "reports" / "drift_report.json"

FEATURES = ["Time", "Amount", "V1", "V2", "V3", "V4", "V14"]
MAX_REFERENCE_ROWS = 10000
MAX_CURRENT_ROWS = 10000


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Prediction database not found. Start the API and "
            "run the transaction simulator first."
        )

    reference_df = pd.read_csv(DATA_PATH)
    reference_df = reference_df.drop(columns=["Class"])

    with sqlite3.connect(DB_PATH) as conn:
        rows = pd.read_sql_query(
            "SELECT features_json FROM predictions "
            "ORDER BY created_at DESC LIMIT ?",
            conn,
            params=(MAX_CURRENT_ROWS,),
        )

    if rows.empty:
        raise RuntimeError("No prediction records available for monitoring.")

    current_df = pd.DataFrame(
        [json.loads(value) for value in rows["features_json"]]
    )

    rng = np.random.default_rng(42)
    reference_sample = reference_df.sample(
        n=min(MAX_REFERENCE_ROWS, len(reference_df)),
        random_state=42,
    )

    report = {
        "reference_rows": len(reference_sample),
        "current_rows": len(current_df),
        "method": "Two-sample Kolmogorov-Smirnov test",
        "features": {},
    }

    drifted_count = 0

    for feature in FEATURES:
        reference = pd.to_numeric(
            reference_sample[feature], errors="coerce"
        ).dropna()

        current = pd.to_numeric(
            current_df[feature], errors="coerce"
        ).dropna()

        if len(reference) < 2 or len(current) < 2:
            report["features"][feature] = {
                "status": "insufficient_data"
            }
            continue

        statistic, p_value = ks_2samp(reference, current)

        # D measures the maximum difference between empirical
        # cumulative distribution functions.
        # 0.10 is a practical effect-size alert threshold,
        # not a universal definition of meaningful drift.
        is_drifted = bool(statistic >= 0.10)

        if is_drifted:
            drifted_count += 1

        report["features"][feature] = {
            "reference_mean": float(reference.mean()),
            "current_mean": float(current.mean()),
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": is_drifted,
        }

    report["drifted_feature_count"] = drifted_count
    report["features_checked"] = len(FEATURES)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nDrift report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
