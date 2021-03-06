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


# Keras / sklearn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


# plot
import matplotlib.pyplot as plt



def load_data():
    data = []
    with open(str(pathlib.Path(__file__).parent.absolute()) + "/data.json", 'r') as f:
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
df = load_data()
top10lb = list(df['lbAlias'].value_counts().head(10).index)
df = df[df['lbAlias'].isin(top10lb)]

df = df.dropna()
df = df[df.caseDesc.apply(lambda x: x != "")]
df = df[df.lbAlias.apply(lambda x: x != "")]

# apply text cleaning function to df['text']
df['caseDesc'] = df['caseDesc'].map(lambda x: clean_text(x))

# print(df['lbAlias'].value_counts())
# print(df.head(10))


# LSTM modelling - RNN
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each legalcase
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 128


tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='0123456789!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, oov_token="<OOV>")
tokenizer.fit_on_texts(df['caseDesc'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# make length of case descriptions equal
X = tokenizer.texts_to_sequences(df['caseDesc'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

y = pd.get_dummies(df['lbAlias']).values
labels = pd.get_dummies(df['lbAlias']).columns.tolist()
print('Shape of label tensor:', y.shape)


# separate training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# get class weights for extremely imbalanced classes
class_weights = class_weight.compute_class_weight('balanced', np.unique(df['lbAlias']), df['lbAlias'])


'''
# SMOTE oversampling
print("Before oversampling: ", X_train.shape)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=4)
train_text_res, train_y_res = sm.fit_resample(X_train, y_train)
print("After oversampling: ", train_text_res.shape)
'''

# create model: RNN --> LSTM
def create_model():
    """
        create model
        https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
    """
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    epochs = 4
    batch_size = 64

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, class_weight=class_weights)

    return model, history


model, history = create_model()


# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print('Accuracy: %f' % (accuracy*100))

'''
from sklearn.metrics import classification_report,confusion_matrix
Y_pred = model.predict(X_test)
y_pred = np.array([np.argmax(pred) for pred in Y_pred])
print('Classification Report:\n', classification_report(y_test, y_pred), '\n')
'''


# create graphics
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  
# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")

model.save(str(pathlib.Path(__file__).parent.absolute()) + "/legalcase_lstm_model.h5")
print("Saved model to disk")
