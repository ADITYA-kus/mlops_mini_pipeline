import json
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
import os

dagshub_token=os.getenv("DVCS3MLFLOW")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri("https://dagshub.com/ADITYA-kus/mlops_mini_pipeline.mlflow/")





def load_model_info(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)



def register_model(model_name: str, model_info: dict):
    model_uri = model_info.get("model_uri")
    if not model_uri:
        # backward compatibility if old json exists
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
    model_version = mlflow.register_model(model_uri, model_name)    
    client= MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="staging",
    )
    
    print("REGISTERING model_uri:", model_uri)
    print(f"✅ Model {model_name} version {model_version.version} registered and push to staging.")




def main():
    model_info_path = os.path.join('reports', 'experiment_info.json')
    model_info = load_model_info(model_info_path)


    register_model("my_logistic_regression_model", model_info)




if __name__ == '__main__':
    main()



