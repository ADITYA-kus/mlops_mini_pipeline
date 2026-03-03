import numpy as np
import pandas as pd
import pickle
import json
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub

# Fix Windows console encoding issue for MLflow emoji output
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


dagshub_token=os.getenv("DVCS3MLFLOW")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri("https://dagshub.com/ADITYA-kus/mlops_mini_pipeline.mlflow/")


# Set up MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/ADITYA-kus/mlops_mini_pipeline.mlflow/")


def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_data(file_path):
    return pd.read_csv(file_path)

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }

def save_json(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4)

def main():
    mlflow.set_experiment("dvc-pipeline1")
    with mlflow.start_run() as run:
        # Paths
        model_path = os.path.join('models', 'bow_model.pkl')
        test_features_path = os.path.join('data', 'features_store', 'testing_feature.csv')
        metrics_path = os.path.join('src', 'modeling', 'metrics.json')
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_info_path = os.path.join(PROJECT_ROOT, "reports", "experiment_info.json")
        # Load model + data
        clf = load_model(model_path)
        test_data = load_data(test_features_path)
        X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

        # Evaluate
        metrics = evaluate_model(clf, X_test, y_test)
        save_json(metrics, metrics_path)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log params
        if hasattr(clf, 'get_params'):
            for k, v in clf.get_params().items():
                mlflow.log_param(k, v)

        # Log model to MLflow under artifact_path="model"
        # Log model to MLflow under artifact_path
        from mlflow.tracking import MlflowClient

       

        # Log model (new API style)
        model_info = mlflow.sklearn.log_model(clf, name="my_logistic_regression_model")

        print("Logged model URI:", model_info.model_uri)

        # Save exact model_uri for registry step (best practice)
        save_json(
            {"run_id": run.info.run_id, "model_uri": model_info.model_uri}, #only change model inbuild uri thab manual and save model uri insted of modelname
            model_info_path
        )


        # Log metrics file as artifact
        mlflow.log_artifact(metrics_path)

        print("✅ Model evaluation and logging complete.")

if __name__ == '__main__':
    main()


    