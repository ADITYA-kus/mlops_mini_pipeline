import numpy as np
import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import yaml

def load_yaml():
    try:
        mx_feature=yaml.safe_load(open('params.yaml','r'))['features_engineerig']['max_features']
        return mx_feature
    except Exception as e:
        print(e)
        raise

# fetch the data from data/processed
def load_processed_data(train_path,test_path):
    try:
        train_data = pd.read_csv(train_path).dropna()
        test_data = pd.read_csv(test_path).dropna()

        return train_data,test_data

    except Exception as e:
        print("error occured during processing data loaading")
        print(e)
        raise


# apply BoW
def convert_array(train_data ,test_data):
    try:
        x_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        
        x_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        return x_train,x_test ,y_train,y_test


    except Exception as e:
        print("an error ocured durig convrting array")
        print(e)
        raise

def apply_bow(x_train,x_test,y_train,y_test,max_feature):
    try:
        vectorizer = TfidfVectorizer(max_features=max_feature)
        x_train_tfidf = vectorizer.fit_transform(x_train)
        
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        
        x_test_tfidf = vectorizer.transform(x_test)
        train_df = pd.DataFrame(x_train_tfidf.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(x_test_tfidf.toarray())
        test_df['label'] = y_test
        return train_df,test_df

    except Exception as e:
        print("an error occured during bag of word")
        print(e)
        raise


def save_features(train_df,test_df):
    try:
        features_dir = os.path.join('data', 'features_store')
        os.makedirs(features_dir, exist_ok=True)
        train_df.to_csv(os.path.join(features_dir, 'training_feature.csv'),index=False)
        test_df.to_csv(os.path.join(features_dir, 'testing_feature.csv'),index=False)

    except Exception as e:
        print("features not saved")
        print(e)
        raise


def main():
    try:
        train_processed_path=os.path.join('data', 'processed', 'train_processed.csv')
        test_processed_path=os.path.join('data', 'processed', 'test_processed.csv')
        train_data,test_data=load_processed_data(train_processed_path,test_processed_path)
        x_train,x_test ,y_train,y_test=convert_array(train_data,test_data)
        
        train_df,test_df=apply_bow( x_train,x_test ,y_train,y_test,load_yaml())
        save_features(train_df,test_df)

    except Exception as e:
        print(e)

        print("featuring enginnering failed")


if __name__=="__main__":
    main()


    