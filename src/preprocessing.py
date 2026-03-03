import pandas as pd
import numpy as np 

import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

def data_load(train_df_path,test_df_path):
    try:
        train_df=pd.read_csv(train_df_path)
        test_df=pd.read_csv(test_df_path)
        return train_df,test_df

    except pd.errors.ParserError as e:
        print(f"unexpected error ocurred during loading test or train csv file{e}")
        raise
    except Exception as e:
        print("failed to load csv file")
        print(e)
        raise    


# transform the data
# nltk.download('wordnet')
# nltk.download('stopwords')
def lemmatization(text):
    try: 
       lemmatizer= WordNetLemmatizer()

       text = text.split()

       text=[lemmatizer.lemmatize(y) for y in text]

       return " " .join(text)
    except Exception as e:
         print("lemitaization error")
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
        print("removing numbers eror")
        print(e)
        raise


def lower_case(text):
    try:

       text = text.split()

       text=[y.lower() for y in text]

       return " " .join(text)
    except Exception as e:
        print("lowwr case eror")
        print(e)
        raise
def removing_punctuations(text):
    ## Remove punctuations
    try:
        text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )
        ## remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text =  " ".join(text.split())
        return text.strip()

    except Exception as e:
        print("panchutaion remove error")
        print(e)
        raise

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        print("removing url error")
        print(e)
        raise

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    try:
        df.content=df.content.apply(lambda content : lower_case(content))
        df.content=df.content.apply(lambda content : remove_stop_words(content))
        df.content=df.content.apply(lambda content : removing_numbers(content))
        df.content=df.content.apply(lambda content : removing_punctuations(content))
        df.content=df.content.apply(lambda content : removing_urls(content))
        df.content=df.content.apply(lambda content : lemmatization(content))
        return df
    except Exception as e:
        print("nomilazation error")
        print(e)
        raise
def save_processed_data(train_processed_data,test_processed_data):
     try:
         train_processed_data.to_csv(os.path.join('data', 'processed', 'train_processed.csv'),index=False)
         test_processed_data.to_csv(os.path.join('data', 'processed', 'test_processed.csv'),index=False)
         print("processed data saved successfully")

     except:
         print('processed data not d saved')
         raise
         
def main():
    try:
        test_path=os.path.join('data', 'raw', 'test_df.csv')
        train_path=os.path.join('data', 'raw', 'train_df.csv')
        train_df,test_df=data_load(train_path,test_path)
        
        train_processed_data = normalize_text(train_df)
        test_processed_data = normalize_text(test_df)
        save_processed_data(train_processed_data,test_processed_data)
        print("preprocessing step complete")
    except:
        print("preprocessing failed")
        raise
        


if __name__=="__main__":
    main()







        
    