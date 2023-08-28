import re, string
import spacy
import joblib
import uvicorn
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from textwrap import wrap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

import sqlite3

#################### transformers for preprocessing ###########################
# Dictionary of English Contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  # print (type(text))
  return [contractions_re.sub(replace, t) for t in text]

def lowercase(text):
    return [t.lower() for t in text]

def removedigits(text):
    return [re.sub('\w*\d\w*','', t) for t in text]

def removepunct(text):
    return [re.sub('[%s]' % re.escape(string.punctuation), '', t) for t in text]

def removespaces(text):
    return [re.sub(' +',' ',t) for t in text]

nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])
def lemmatize(text):
    return [' '.join([token.lemma_ for token in list(nlp(t)) if (token.is_stop==False)]) for t in text]
#################### transformers for preprocessing ###########################

# load model
model = joblib.load("gbPipelineTfidf.pkl")

# create request model
class Data(BaseModel):
    data: list

# create app
app = FastAPI()

# make connection to sqlite db
con = sqlite3.connect("sklearn.db")
cur = con.cursor()
res = cur.execute("SELECT name FROM sqlite_master WHERE name='sklearn'")
# create table if not already created
if res.fetchone() is None:
    cur.execute("CREATE TABLE sklearn(id INTEGER, text TEXT, inference INTEGER)")



@app.get("/sklearn/infer")
async def sklearn_infer(data: Data):
    # fetch request data
    data = data.data
    
    # get inference table fields
    id = hash(data[0])
    text = data[0]
    inference = model.predict(data).tolist()[0]
    
    # insert into inference table
    cur.execute(f"""INSERT INTO sklearn VALUES({id}, "{text}", {inference})""")
    
    # make data persist
    con.commit()
    
    # return response   
    return JSONResponse({"inference": inference})

# run app
uvicorn.run(app, host="0.0.0.0")