import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
import yaml

def load_yaml():
    try:
       params= yaml.safe_load(open(r'params.yaml','r'))['model_building']
       return params
    except Exception as e:
        print(e)
        raise

def load_training_features(train_path):
    try:
        train_feature = pd.read_csv(train_path)
        X_train = train_feature.iloc[:, :-1].values
        y_train = train_feature.iloc[:, -1].values
        return X_train, y_train
    except Exception as e:
        print(f"Error loading training features from {train_path}: {e}")
        raise

def model_training(x_train, y_train,params):
    try:
        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])
        clf.fit(x_train, y_train)
        return clf
    except Exception as e:
        print(f"Model training failed: {e}")
        raise

def save_model(clf):
    try:
        os.makedirs('models', exist_ok=True)
        with open('models/bow_model.pkl', 'wb') as f:
            pickle.dump(clf, f)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Failed to save model: {e}")
        raise

def main():
    try:
        train_features_path = 'data/features_store/training_feature.csv'
        x_train, y_train = load_training_features(train_features_path)
        param=load_yaml()
        clf = model_training(x_train, y_train,param)
        save_model(clf)
        print("Model building stage complete.")
    except Exception as e:
        print(f"Model building stage failed: {e}")
        raise

if __name__ == "__main__":
    main()  