
# import libraties
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# NLTK libraries
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

#Using Cosine Similarity
from sklearn.metrics.pairwise import pairwise_distances

# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')

#Modelling 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import xgboost as xgb  # Load this xgboost

from collections import Counter
from imblearn.over_sampling import SMOTE
#from sklearn.externals import joblib
import joblib
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)  # intitialize the flaks app  # common 

#with open('recommend_model.pkl','rb') as fp:
#	recommend_model = pickle.load(fp)

recommend_model = joblib.load('recommend_model_joblib.pkl')

with open ('sentiment_model.pkl','rb') as f:
    #unpickler = pickle.Unpickler(f)
    sentiment_model = pickle.load(f)
    
with open ('lemmatize_sentence.pkl','rb') as f:
    lemmatize_sentence = pickle.load(f)

with open ('word_vectorizer.pkl','rb') as f:
    word_vectorizer = pickle.load(f)


# Reading data from the the file 
data = pd.read_csv('sample30.csv' , encoding='latin-1')

#render_template('index.html')

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict',methods=['POST'])
def recommend_pred():
    user_input = request.form.get('user_name')
    temp = pd.DataFrame(recommend_model.loc[user_input].sort_values(ascending=False))
    if temp.shape[0] > 20:    
        top20 =  temp[0:20]
    else:
        top20 =  temp
    top20 = top20.reset_index()
    top20.rename(columns = {user_input:'score'}, inplace = True)
    top20.insert(2, "Positive Sentiment(%)", "") 
    for prod in top20['name']:    
        #print(prod)
        rev = data[data['name'] == prod]
        if rev.shape[0] > 0:
            temp = rev['reviews_text'].apply(lambda x: lemmatize_sentence(x))
            temp1 = word_vectorizer.transform(temp)
            temp2 = sentiment_model.predict(temp1)
            pos = sum(temp2)
            total = len(temp2)
            percent = round(pos*100/total,2)
            top20.loc[top20['name'] == prod, ['Positive Sentiment(%)']] = percent

    
    temp = top20.sort_values(by="Positive Sentiment(%)",ascending=False)#[0:5]
    if temp.shape[0] > 5:
        top5 =  temp[0:5]
    else:
        top5 =  temp
    #top5.shape
    top5['score'] = round(top5['score'],2)
    
    return  render_template('index.html',tables=[top5.to_html()], titles = top5.columns.values)


# Any HTML template in Flask App render_template

if __name__ == '__main__' :
    app.run(debug=True )  # this command will enable the run of your flask app or api
    
    #,host="0.0.0.0")







