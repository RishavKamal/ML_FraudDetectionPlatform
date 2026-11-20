# Real-Time Fraud Detection & Model Monitoring Platform

## Overview

An end-to-end machine learning project that detects potentially fraudulent credit card transactions and provides a monitoring dashboard for model predictions, inference latency, and input data drift.

The project uses a real historical Kaggle dataset and simulates real-time inference by replaying transactions through a local REST API.

> **Important:** This is an educational prototype, not a production banking system or a connection to a live payment gateway.

## Features

- Fraud detection using Scikit-learn
- Comparison of Logistic Regression and Random Forest
- Imbalanced classification evaluation
- Validation-based classification threshold
- FastAPI prediction endpoint
- SQLite prediction logging
- Streamlit monitoring dashboard
- Automated transaction simulator
- Statistical input-drift report using the Kolmogorov–Smirnov (KS) test
- Automated API tests using Pytest

## Dataset

**ULB Credit Card Fraud Detection**

Dataset: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains:

- **Total transactions:** 284,807
- **Fraudulent transactions:** 492
- **Features:** 30
- **Target column:** `Class`

The dataset includes anonymized PCA components, transaction time, and transaction amount.

The target column `Class` represents:

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

### Dataset Setup

1. Download the dataset from Kaggle.
2. Extract the downloaded file.
3. Place `creditcard.csv` inside the following directory:

```text
data/raw/creditcard.csv
```

The project uses this dataset for model training, evaluation, transaction simulation, and monitoring.

## Model Evaluation

The initial training run compared Logistic Regression and Random Forest.

Random Forest was selected using validation average precision.

### Held-Out Test Results

| Metric | Result |
|---|---:|
| Average Precision | 0.81705 |
| ROC-AUC | 0.95902 |
| Precision | 0.94828 |
| Recall | 0.74324 |
| F1-score | 0.83333 |

**Selected model:** Random Forest

**Validation-selected decision threshold:** Approximately `0.619662`

The decision threshold was selected using the validation set.

These results come from one stratified random split. They do not establish performance on future transactions, and the model requires additional validation before any real-world use.

## System Architecture

The project follows an end-to-end machine learning workflow.

```text
                 Historical Kaggle Dataset
                          |
                          v
                  Model Training Pipeline
                          |
             +------------+------------+
             |                         |
             v                         v
      Logistic Regression        Random Forest
             |                         |
             +------------+------------+
                          |
                          v
                Model Evaluation
                          |
                          v
                 Selected Model
                          |
                          v
                  Saved Model Artifact
                          |
                          v
                  FastAPI Prediction API
                          |
             +------------+------------+
             |                         |
             v                         v
       Transaction Input         SQLite Database
             |                  Prediction Logging
             v
       Fraud Prediction
             |
             v
       Prediction Response
             |
             v
      Streamlit Dashboard
             |
             v
       Monitoring & Metrics


       Historical Transactions
                 |
                 v
       Transaction Simulator
                 |
                 v
          FastAPI Prediction
                 |
                 v
         Logged Transactions
                 |
                 v
          Drift Monitoring
                 |
                 v
           Drift Report
```

### Architecture Components

1. **Dataset:** Supplies historical credit card transactions.
2. **Training pipeline:** Preprocesses data, trains candidate models, evaluates them, and saves the selected model.
3. **FastAPI:** Loads the saved model and provides transaction predictions through a REST API.
4. **SQLite:** Stores prediction records and inference metadata.
5. **Transaction simulator:** Replays historical transactions through the API to simulate real-time inference.
6. **Streamlit dashboard:** Summarizes prediction activity, latency, and available evaluation metrics.
7. **Drift monitor:** Compares reference and logged transaction feature distributions.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Comparison | Logistic Regression, Random Forest |
| API Backend | FastAPI |
| API Server | Uvicorn |
| Database | SQLite |
| Dashboard | Streamlit |
| Drift Detection | Kolmogorov–Smirnov Test |
| Model Persistence | Joblib |
| API Testing | Pytest |

## Project Structure

```text
Real-Time-Fraud-Detection/
│
├── data/
│   └── raw/
│       └── creditcard.csv
│
├── models/
│   └── fraud_model.joblib
│
├── reports/
│   └── simulation_results.csv
│
├── src/
│   ├── train.py
│   ├── api.py
│   ├── dashboard.py
│   ├── simulate_transactions.py
│   └── drift_monitor.py
│
├── tests/
│   └── API test files
│
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: The model artifact, simulation report, and database are generated during project execution. Their exact filenames or locations may depend on the implementation.*

## Setup and Installation

### 1. Clone the Repository

```powershell
git clone <your-github-repository-url>
```

Navigate into the project directory:

```powershell
cd Real-Time-Fraud-Detection
```

### 2. Create a Virtual Environment

```powershell
py -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use the following command for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Download the Dataset

Download the ULB Credit Card Fraud Detection dataset from:

[Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place the extracted `creditcard.csv` file inside:

```text
data/raw/creditcard.csv
```

## Running the Project

Run each component in a separate terminal.

### Step 1: Train the Model

```powershell
python src/train.py
```

The training pipeline:

1. Loads the historical dataset.
2. Prepares the features and target.
3. Splits the dataset into training, validation, and test sets.
4. Trains Logistic Regression and Random Forest.
5. Evaluates the candidate models.
6. Selects a model using validation average precision.
7. Selects a classification threshold using validation performance.
8. Evaluates the selected model on the held-out test set.
9. Saves the selected model artifact.

Run training before starting the API if the model artifact does not already exist.

### Step 2: Start the FastAPI Backend

Open a new terminal in the project directory.

```powershell
uvicorn src.api:app --reload
```

The API runs locally at:

```text
http://127.0.0.1:8000
```

### Step 3: Open API Documentation

Open the following URL in your browser:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

The Swagger documentation allows you to:

- View available API endpoints.
- Submit a transaction for prediction.
- Inspect the prediction response.
- Test the API without writing a separate client.

### Step 4: Start the Streamlit Dashboard

Open another terminal.

```powershell
streamlit run src/dashboard.py
```

Dashboard URL:

[http://localhost:8501](http://localhost:8501)

The dashboard provides a local interface for reviewing prediction activity and available monitoring information.

### Step 5: Run the Transaction Simulator

Open another terminal.

```powershell
python src/simulate_transactions.py --count 100
```

The simulator:

1. Loads historical transactions.
2. Selects transactions for simulation.
3. Removes the actual `Class` label before sending the transaction.
4. Sends transactions to the FastAPI prediction endpoint.
5. Collects prediction responses.
6. Saves simulation results.

To simulate 10 transactions:

```powershell
python src/simulate_transactions.py --count 10
```

To simulate 50 transactions with a delay:

```powershell
python src/simulate_transactions.py --count 50 --delay 0.1
```

The simulation results are saved as:

```text
reports/simulation_results.csv
```

### Step 6: Generate the Drift Report

```powershell
python src/drift_monitor.py
```

The drift monitor compares reference transaction features against logged transaction features and generates a statistical drift report.

### Step 7: Run Automated Tests

```powershell
python -m pytest tests -v
```

Pytest executes the available automated API tests.

## API Endpoints

The FastAPI backend provides the following endpoints.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check API health |
| `POST /predict` | Predict whether a transaction is potentially fraudulent |
| `GET /metrics/summary` | Retrieve prediction summary metrics |
| `GET /predictions?limit=10` | Retrieve recent prediction records |

### Prediction Input

The prediction endpoint expects transaction features inside the `transaction` object.

The input contains:

- `Time`
- `V1` through `V28`
- `Amount`

The `Class` target label must not be included in the prediction request.

Example structure:

```json
{
  "transaction": {
    "Time": 0,
    "V1": 0.0,
    "V2": 0.0,
    "V3": 0.0,
    "V4": 0.0,
    "V5": 0.0,
    "V6": 0.0,
    "V7": 0.0,
    "V8": 0.0,
    "V9": 0.0,
    "V10": 0.0,
    "V11": 0.0,
    "V12": 0.0,
    "V13": 0.0,
    "V14": 0.0,
    "V15": 0.0,
    "V16": 0.0,
    "V17": 0.0,
    "V18": 0.0,
    "V19": 0.0,
    "V20": 0.0,
    "V21": 0.0,
    "V22": 0.0,
    "V23": 0.0,
    "V24": 0.0,
    "V25": 0.0,
    "V26": 0.0,
    "V27": 0.0,
    "V28": 0.0,
    "Amount": 100.0
  }
}
```

> This is an illustrative input structure, not an actual transaction from the dataset. Use a real dataset row or Swagger's request schema when testing predictions.

## Monitoring Interpretation

### Prediction Volume

The number of transactions processed by the local API.

### Flagged Transactions

Transactions classified as potentially fraudulent using the configured decision threshold.

A flagged transaction is a model decision, not proof of fraud.

### Inference Latency

The time spent calculating a prediction.

This measurement excludes the full HTTP round trip.

### Input Drift

Statistical differences between reference and current input feature distributions.

Input drift does not automatically imply declining model accuracy.

### Model Performance

Precision, recall, and other classification metrics require actual transaction labels.

Predictions alone cannot establish whether a transaction was genuinely fraudulent.

## Limitations

- Inference is simulated using historical transactions.
- The dataset contains anonymized features rather than full real-world transaction details.
- The current drift monitor is a lightweight statistical implementation.
- No automated retraining is included.
- No distributed streaming infrastructure is included.
- No production authentication is included.
- SQLite is intended for this local demonstration, not high-throughput production deployment.
- The initial evaluation uses a random stratified split rather than a time-based validation design.
- Model performance on historical data does not guarantee performance on future transactions.

## Future Improvements

- Time-based model evaluation
- MLflow experiment tracking
- Population-level drift dashboards
- Delayed ground-truth label ingestion
- Model versioning and rollback
- Docker Compose deployment
- CI/CD with GitHub Actions
- Authentication and authorization
- Monitoring alerts
- Production observability
- Automated model retraining with appropriate validation

## License and Dataset Attribution

The dataset is provided by the ULB Credit Card Fraud Detection project through Kaggle.

Dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Review the dataset's current terms and attribution requirements on Kaggle before redistributing the data.

Do not commit the raw dataset, credentials, API keys, or other sensitive information to the repository.

---

**Project:** Real-Time Fraud Detection & Model Monitoring Platform

**Purpose:** Educational machine learning, API development, transaction simulation, and model monitoring.
