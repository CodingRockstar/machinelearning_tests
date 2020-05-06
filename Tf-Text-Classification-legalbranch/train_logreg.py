# train a model for text classification with tensorflow / keras
from __future__ import absolute_import, division, print_function

import re
import json
import pathlib

import string
import numpy as np
import pandas as pd  # pip install pandas


# nltk
import nltk  # pip install --user -U nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# Keras / sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split



def load_data(filename):
    data = []
    with open(str(pathlib.Path(__file__).parent.absolute()) + "/" + filename, 'r') as f:
        imported = json.load(f)
        data = pd.DataFrame(imported['data'])
        data.drop(['id', 'caseTitle', 'lbName', 'ltName', 'ltAlias'], axis=1, inplace=True)

    return data

### Text Normalizing function. Part of the following function was taken from this link. 
### https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b
def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("german"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=äöüß]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('german')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text



# load data and remove null-Values
df = load_data("data.json")
top10lb = list(df['lbAlias'].value_counts().head(10).index)
df = df[df['lbAlias'].isin(top10lb)]

# print(top10lb)
# print(filteredDf.head(20))

df = df.dropna()
df = df[df.caseDesc.apply(lambda x: x != "")]
df = df[df.lbAlias.apply(lambda x: x != "")]

# apply text cleaning function to df['text']
df['text'] = df['caseDesc'].map(lambda x: clean_text(x))
tags = pd.get_dummies(df['lbAlias']).columns.tolist()

# print(df['lbAlias'].value_counts())
# print(df.head(10))


X = df.text
y = df.lbAlias
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


# naive bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
""" 
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, labels=tags, zero_division=0)) """





# support vector machine (SVM)
""" from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)


y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, labels=tags, zero_division=0)) """




# logistic regression
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5, max_iter=500)),
               ])
# train model
logreg.fit(X_train, y_train)

# test model
y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, labels=tags, zero_division=0))



# make validation predictions here
validation_data = load_data("legalcase_validation.json")

# filter by known legal branches
validation_data = validation_data[validation_data['lbAlias'].isin(top10lb)]

# sanitize text
validation_data['caseDesc'] = validation_data['caseDesc'].map(lambda x: clean_text(x))

for index, row in validation_data.iterrows():
    predictions = logreg.predict([row['caseDesc']])
    print("Legal branch predicted: {} (Soll: {})".format(predictions[0], row['lbAlias']))
