
from pathlib import Path
import argparse
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
OUTPUT_PATH = ROOT / "reports" / "simulation_results.csv"

API_URL = "http://127.0.0.1:8000/predict"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--url", default=API_URL)
    args = parser.parse_args()

    if args.count < 1 or args.count > 10000:
        parser.error("--count must be between 1 and 10000")

    if args.delay < 0:
        parser.error("--delay cannot be negative")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # The Class column is retained locally for evaluation,
    # but is never sent to the prediction API.
    df = pd.read_csv(DATA_PATH)
    df = df.sample(
        n=min(args.count, len(df)),
        random_state=42,
    )

    results = []

    print(f"Sending {len(df)} transactions to {args.url}")

    for index, row in df.iterrows():
        actual_class = int(row["Class"])
        features = row.drop(labels=["Class"]).to_dict()

        try:
            response = requests.post(
                args.url,
                json={"transaction": features},
                timeout=15,
            )
            response.raise_for_status()
            prediction = response.json()

            results.append({
                "dataset_row": int(index),
                "actual_class": actual_class,
                "predicted_class": prediction["predicted_class"],
                "fraud_probability": prediction["fraud_probability"],
                "latency_ms": prediction["latency_ms"],
                "prediction_id": prediction["prediction_id"],
            })

            print(
                f"Row {index}: "
                f"actual={actual_class}, "
                f"predicted={prediction['predicted_class']}, "
                f"probability={prediction['fraud_probability']:.4f}"
            )

        except requests.RequestException as exc:
            print(f"Request failed for row {index}: {exc}")

        if args.delay:
            time.sleep(args.delay)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError(
            "No successful predictions. Check whether FastAPI is running."
        )

    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSuccessful predictions: {len(results_df)}")
    print(f"Results saved to: {OUTPUT_PATH}")

    if len(results_df) != args.count:
        print("Warning: Some transactions failed; inspect the errors above.")

    print("\nSimulation summary:")
    print(f"Actual fraud cases: {results_df['actual_class'].sum()}")
    print(f"Predicted fraud cases: {results_df['predicted_class'].sum()}")


if __name__ == "__main__":
    main()