import json
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("https://dagshub.com/ADITYA-kus/mlops_mini_pipeline.mlflow/")
dagshub.init(repo_owner='ADITYA-kus', repo_name='mlops_mini_pipeline', mlflow=True)

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
        stage="production",
    )
    
    print("REGISTERING model_uri:", model_uri)
    print(f"✅ Model {model_name} version {model_version.version} registered and push to production.")




def main():
    model_info_path = 'reports/experiment_info.json'
    model_info = load_model_info(model_info_path)


    register_model("my_logistic_regression_model", model_info)




if __name__ == '__main__':
    main()

