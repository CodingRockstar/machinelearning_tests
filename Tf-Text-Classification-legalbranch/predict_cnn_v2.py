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
from keras import utils
from sklearn.preprocessing import LabelBinarizer, LabelEncoder



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


train_size = int(len(df) * .8)
train_posts = df['caseDesc'][:train_size]
train_tags = df['lbAlias'][:train_size]

test_posts = df['caseDesc'][train_size:]
test_tags = df['lbAlias'][train_size:]

max_words = 25000
tokenize = Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts) # only fit on train

x_train = tokenize.texts_to_matrix(train_posts)

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
labels = encoder.classes_

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)


# load / initiate model
model = load_model(str(pathlib.Path(__file__).parent.absolute()) + "/legalcase_cnn_model.h5")



# make preditction on this model
validation_data = load_data("legalcase_validation.json")

# filter by known legal branches
validation_data = validation_data[validation_data['lbAlias'].isin(top10lb)]

# sanitize text
validation_data['caseDesc'] = validation_data['caseDesc'].map(lambda x: clean_text(x))

for index, row in validation_data.iterrows():
    seq = tokenize.texts_to_matrix([row['caseDesc']])
    predictions = model.predict(seq)
    pred = predictions[0]
    print("Legal branch predicted: {} (Soll: {}) with {}".format(labels[np.argmax(pred)], row['lbAlias'], pred[np.argmax(pred)]))
    
