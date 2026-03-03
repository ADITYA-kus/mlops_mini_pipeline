import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml
def load_yaml():
    try:
      param=yaml.safe_load(open('params.yaml','r'))
      test_siz=param['data_ingestion']['test_size']
      return test_siz
    except Exception as e:
      print(e)
      raise  
   
def data_ingestion(data_url):
    try:
      raw_data=pd.read_csv(data_url)
      return raw_data
    except pd.errors.ParserError as e:
       print(f"failed to parse the csv file from url{data_url}")
       print(e)
       raise
    except Exception as e:
       print(e)
       raise

   
def basic_data_process(raw_data):
   try:
      raw_data
      raw_data.dropna(inplace=True)
      raw_data.drop(columns=["tweet_id"],inplace=True)
      raw_data=raw_data[raw_data['sentiment'].isin(['love','sadness'])].copy()
      raw_data['sentiment']=raw_data['sentiment'].map({'sadness':0,'love':1}).astype('int64')
      return raw_data
   except Exception as e:
      print(f"An unexpected error occured during basic preprocesing {e}")
      raise
   
def save_external_data(final_df):
    try:
      external_dir = os.path.join('data', 'external')
      os.makedirs(external_dir, exist_ok=True)
      final_df.to_csv(os.path.join(external_dir, 'external_data.csv'))
    except Exception as e:
      print(f"an unexpected error occured during external data saving")
      print(e)
      raise  


def save_data(train_df,test_df):
    try:
      raw_dir = os.path.join('data', 'raw')
      os.makedirs(raw_dir, exist_ok=True)
      train_df.to_csv(os.path.join(raw_dir, 'train_df.csv'),index=False)
      test_df.to_csv(os.path.join(raw_dir, 'test_df.csv'),index=False)
    except Exception as e:
      print('An unexcepted error occured during train test data saving in raw file')
      print(e)
      raise      
   

def start():
    try:
        data_url="https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
        df=data_ingestion(data_url)
        final_data=basic_data_process(df)
        save_external_data(final_data)
        test_siz=load_yaml()
        train_df,test_df=train_test_split(final_data,test_size=test_siz,random_state=42)
        save_data(train_df,test_df)
    except Exception as e:
       print(f"error{e}")
       print("failed to data ingestion process")

if __name__=="__main__":
    start()

      

   