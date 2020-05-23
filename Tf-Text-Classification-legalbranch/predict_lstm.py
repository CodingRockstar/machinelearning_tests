# train a model for text classification with tensorflow / keras
from __future__ import absolute_import, division, print_function

import re
import json
import pathlib

import string
import numpy as np
import pandas as pd


# nltk
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model



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

df = df.dropna()
df = df[df.caseDesc.apply(lambda x: x != "")]
df = df[df.lbAlias.apply(lambda x: x != "")]

# apply text cleaning function to df['text']
df['caseDesc'] = df['caseDesc'].map(lambda x: clean_text(x))


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each legalcase
MAX_SEQUENCE_LENGTH = 128


tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='0123456789!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, oov_token="<OOV>")
tokenizer.fit_on_texts(df['caseDesc'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# make length of case descriptions equal
X = tokenizer.texts_to_sequences(df['caseDesc'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')

Y = pd.get_dummies(df['lbAlias']).values
labels = pd.get_dummies(df['lbAlias']).columns.tolist()

# load / initiate model
model = load_model(str(pathlib.Path(__file__).parent.absolute()) + "/legalcase_lstm_model.h5")



# make preditction on this model
validation_data = load_data("legalcase_validation.json")

# filter by known legal branches
validation_data = validation_data[validation_data['lbAlias'].isin(top10lb)]

# sanitize text
validation_data['caseDesc'] = validation_data['caseDesc'].map(lambda x: clean_text(x))

result = []
for index, row in validation_data.iterrows():
    seq = tokenizer.texts_to_sequences([row['caseDesc']])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
    predictions = model.predict(padded)
    pred = predictions[0]
    result.append([labels[np.argmax(pred)], row['lbAlias'], pred[np.argmax(pred)], (labels[np.argmax(pred)] == row['lbAlias'])])
    
rPd = pd.DataFrame(result, columns=['predicted', 'given', 'accuracy', 'result'])
print("\n# of correct predictions: {}".format(len(rPd[(rPd.result == True)])))
print("# of incorrect predictions: {}\n".format(len(rPd[(rPd.result == False)])))
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):  # more options can be specified also
    print(rPd)