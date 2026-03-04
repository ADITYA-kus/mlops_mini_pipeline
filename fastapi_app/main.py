import pandas as pd
import numpy as np
from fastapi import FastAPI,Form,Request
import pickle
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
app=FastAPI()
from fastapi.responses import HTMLResponse

import mlflow 
from mlflow.tracking import MlflowClient
import dagshub

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import os
nltk.download('stopwords')
nltk.download('wordnet')    
dagshub_token=os.getenv("DVCS3MLFLOW")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri("https://dagshub.com/ADITYA-kus/mlops_mini_pipeline.mlflow/")










def lower_case(text):
    try:
        return text.lower()
    except Exception as e:
        print("lower case error")
        print(e)
        raise

def remove_stop_words(text):
    try:
       stop_words = set(stopwords.words("english"))
       Text=[i for i in str(text).split() if i not in stop_words]
       return " ".join(Text)

    except Exception as e:
        print("remove stopword error")
        print(e)
        raise    

def removing_numbers(text):
    try:
         text=''.join([i for i in text if not i.isdigit()])
         return text

    except Exception as e:
        print("remove number error")
        print(e)
        raise

def removing_punctuations(text):
    try:
        punctuations = '''!()-[]{};:'"<>./?@#$%^&*_~'''
        text=''.join([i for i in text if i not in punctuations])
        return text

    except Exception as e:
        print("remove punctuation error")
        print(e)
        raise   


def removing_urls(text):
    try:
        import re
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        print("remove url error")
        print(e)
        raise

def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
    except Exception as e:
        print("lemmatization error")
        print(e)
        raise       


def normalized_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text



templates = Jinja2Templates(directory="templates")




app = FastAPI(title="Sentiment Analysis API")

@app.get('/',response_class=HTMLResponse)# fastapi by default returns json response but we want to return html response so we need to specify the response class
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request,"result": None})





def get_latest_model_version(model_name: str) -> str | None:
    """
    Prefer Production stage.
    If none, fall back to the newest available version among all stages.
    """
    client = MlflowClient()

    # 1) Try Production first
    prod = client.get_latest_versions(model_name, stages=["Production"])
    if prod:
        return prod[0].version

    # 2) Fall back: get ALL versions and pick highest version number
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None

    # versions are strings; convert to int for sorting
    latest = max(versions, key=lambda v: int(v.version))
    return latest.version


MODEL_NAME = "my_logistic_regression_model"
model_version = get_latest_model_version(MODEL_NAME)
if not model_version:
    raise RuntimeError(
        f"No versions found in MLflow registry for model '{MODEL_NAME}'. "
        "Register a model version first."
    )

model_uri = f"models:/{MODEL_NAME}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)






base_dir = os.path.dirname(os.path.abspath(__file__))   # current script folder
parent_dir = os.path.dirname(base_dir)                  # go up one level
file_path = os.path.join( "models", "vectorizer.pkl")

try:

    with open(file_path,'rb') as f:
        vectorizer_model=pickle.load(f)
except Exception as e:
    print("Error loading vectorizer model:")
    print(e)
    raise








@app.post('/predict',response_class=HTMLResponse)
def predict_sentiment(request:Request, text: str = Form(...)):
    cleaned_text=normalized_text(text)


    text_vector=vectorizer_model.transform([cleaned_text])       
  

     
    prediction=model.predict(text_vector.toarray())[0]
    sentiment = "Positive" if prediction == 1 else "Negative"

    return templates.TemplateResponse("index.html", {"request": request, "result":f"{prediction} ({sentiment})"})




