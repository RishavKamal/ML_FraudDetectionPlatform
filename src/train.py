"""Train and evaluate the fraud-detection model.

Pipeline:
1. Load and validate the dataset.
2. Split it into training, validation, and test sets.
3. Train Logistic Regression and Random Forest candidates.
4. Select the model and decision threshold using validation data only.
5. Evaluate the selected model once on held-out test data.
6. Save the model artifact and evaluation reports for the API/dashboard.
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------- Project configuration -------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
TEST_PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"

RANDOM_STATE = 42
TARGET = "Class"
EXPECTED_PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
REQUIRED_COLUMNS = ["Time", *EXPECTED_PCA_FEATURES, "Amount", TARGET]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Read the CSV and check that it has the expected usable structure."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Place creditcard.csv inside the project's data/raw/ folder."
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The dataset is empty.")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Use only the known input columns and the target; extra CSV columns are ignored.
    df = df[REQUIRED_COLUMNS].copy()

    if df.isnull().any().any():
        columns_with_missing = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Missing values found in columns: {columns_with_missing}")

    # Reject non-numeric values instead of silently passing them into the model.
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("The dataset contains infinite or non-finite values.")

    target_values = set(df[TARGET].unique())
    if target_values != {0, 1}:
        raise ValueError(
            f"Expected Class to contain both 0 and 1 only; found {sorted(target_values)}."
        )

    if df.duplicated().any():
        duplicate_count = int(df.duplicated().sum())
        print(
            f"Warning: found {duplicate_count:,} exact duplicate rows. "
            "They are retained to keep this training run consistent with the dataset."
        )

    return df


def split_data(df: pd.DataFrame):
    """Create stratified 70%/15%/15% train, validation, and test splits."""
    X = df.drop(columns=TARGET)
    y = df[TARGET].astype(int)

    # First separate 70% for training and 30% for validation plus testing.
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split the remaining 30% equally to get 15% validation and 15% test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )

    print("\nDataset split (stratified):")
    for split_name, labels in (
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ):
        print(
            f"{split_name:10} {len(labels):>7,} rows | "
            f"fraud={int(labels.sum()):>4,} | rate={labels.mean():.4%}"
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_pipeline(classifier) -> Pipeline:
    """Scale Time and Amount, then fit the requested classifier.

    Scaling is useful for Logistic Regression. Keeping preprocessing inside a
    Pipeline ensures it is fitted only on training data and reused at inference.
    Random Forest does not require scaling, but can safely use the same pipeline.
    """
    preprocessing = ColumnTransformer(
        transformers=[("scale_time_amount", StandardScaler(), ["Time", "Amount"])],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def get_candidate_models() -> dict:
    """Return the candidate models with fixed settings for reproducibility."""
    return {
        "logistic_regression": make_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
    }


def train_and_select_model(X_train, y_train, X_val, y_val):
    """Train candidates and select the one with highest validation Average Precision."""
    best_name = None
    best_model = None
    best_average_precision = -1.0
    comparison = {}

    for name, model in get_candidate_models().items():
        print(f"\nTraining {name}...")
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start_time

        # Average Precision evaluates the precision-recall curve and is useful
        # when fraud is rare compared with legitimate transactions.
        probabilities = model.predict_proba(X_val)[:, 1]
        average_precision = average_precision_score(y_val, probabilities)

        comparison[name] = {
            "average_precision": float(average_precision),
            "training_seconds": round(training_seconds, 2),
        }
        print(f"Validation Average Precision: {average_precision:.5f}")
        print(f"Training time: {training_seconds:.2f} seconds")

        if average_precision > best_average_precision:
            best_name = name
            best_model = model
            best_average_precision = average_precision

    if best_model is None:
        raise RuntimeError("No model was successfully trained.")

    print(f"\nSelected model: {best_name}")
    return best_name, best_model, comparison


def choose_threshold(model, X_val, y_val) -> float:
    """Choose the threshold with the best validation F1-score.

    The held-out test set is not used to choose this threshold. This keeps the
    final test evaluation separate from model-selection decisions.
    """
    probabilities = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probabilities)

    if thresholds.size == 0:
        print("Warning: no thresholds returned; using 0.5.")
        return 0.5

    # precision and recall contain one extra final point compared with thresholds.
    denominator = precision[:-1] + recall[:-1]
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    # A threshold of zero can label every transaction as fraud, so exclude it
    # from the search when positive thresholds are available.
    valid_indices = np.flatnonzero(thresholds > 0)
    if valid_indices.size == 0:
        print("Warning: no positive threshold found; using 0.5.")
        return 0.5

    best_index = valid_indices[np.argmax(f1_scores[valid_indices])]
    selected_threshold = float(thresholds[best_index])

    print(f"Validation-selected threshold: {selected_threshold:.6f}")
    print(f"Validation F1 at threshold: {f1_scores[best_index]:.5f}")
    return selected_threshold


def evaluate_model(model, threshold, X_test, y_test):
    """Evaluate the selected model on the untouched test split."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    metrics = {
        "test_average_precision": float(average_precision_score(y_test, probabilities)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["legitimate", "fraud"],
            output_dict=True,
            zero_division=0,
        ),
    }

    print("\nHeld-out test results")
    print("---------------------")
    for name in (
        "test_average_precision",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "threshold",
    ):
        print(f"{name}: {metrics[name]:.5f}")

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("                 Predicted 0   Predicted 1")
    print(f"Actual 0         {matrix[0, 0]:>10}   {matrix[0, 1]:>12}")
    print(f"Actual 1         {matrix[1, 0]:>10}   {matrix[1, 1]:>12}")

    return metrics, probabilities, predictions


def save_outputs(
    model_name,
    model,
    threshold,
    metrics,
    validation_comparison,
    X_test,
    y_test,
    probabilities,
    predictions,
):
    """Save the model artifact and reports in the format expected by the API."""
    # Keep these keys stable: src/api.py expects model, threshold, model_name,
    # and feature_names in this Joblib artifact.
    artifact = {
        "model": model,
        "threshold": float(threshold),
        "model_name": model_name,
        "feature_names": list(X_test.columns),
    }
    joblib.dump(artifact, MODEL_PATH)

    report = {
        "dataset": "Kaggle ULB Credit Card Fraud Detection",
        "model_name": model_name,
        "validation_model_comparison": validation_comparison,
        "test_metrics": metrics,
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    test_predictions = pd.DataFrame(
        {
            "actual_class": y_test.to_numpy(),
            "fraud_probability": probabilities,
            "predicted_class": predictions,
        }
    )
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    print("\nSaved outputs:")
    print(f"Model artifact: {MODEL_PATH}")
    print(f"Evaluation report: {METRICS_PATH}")
    print(f"Test predictions: {TEST_PREDICTIONS_PATH}")


def main() -> None:
    """Run the full training, validation, testing, and saving workflow."""
    df = load_data()
    print(f"Total transactions: {len(df):,}")
    print(f"Total frauds: {int(df[TARGET].sum()):,}")
    print(f"Fraud prevalence: {df[TARGET].mean():.4%}")
    print(f"Input feature count: {len(REQUIRED_COLUMNS) - 1}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    model_name, model, comparison = train_and_select_model(
        X_train, y_train, X_val, y_val
    )
    threshold = choose_threshold(model, X_val, y_val)
    metrics, probabilities, predictions = evaluate_model(
        model, threshold, X_test, y_test
    )
    save_outputs(
        model_name,
        model,
        threshold,
        metrics,
        comparison,
        X_test,
        y_test,
        probabilities,
        predictions,
    )


if __name__ == "__main__":
    main()
