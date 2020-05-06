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
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout


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
EMBEDDING_DIM = 100


tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='0123456789!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['caseDesc'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# make length of case descriptions equal
X = tokenizer.texts_to_sequences(df['caseDesc'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['lbAlias']).values
labels = pd.get_dummies(df['lbAlias']).columns.tolist()
print('Shape of label tensor:', Y.shape)


# separate training and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# create model: RNN --> LSTM
def create_model():
    """
        create model
        https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
    """
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    epochs = 4
    batch_size = 64

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    return model, history


model, history = create_model()


# evaluate model
accr = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))


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



# make preditction on this model
new_complaint = [
    'Ich habe mich vor 4 Jahren aufgrund narzisstischen Missbrauches und seinem Drogenkonsum von meinem Exmann getrennt. Seitdem zerrt er mich und die Kinder immer wieder vor Gericht, bringt sie und mich immer wieder in missliche und äußerst belastende Situationen. Er schreckt auch nicht vor Straftaten wie Stalking, Einbruch, Datenmissbrauch, Diskreditieren und Diebstahl etc. zurück. Nach außen setzt er eine Maske auf, manipuliert und instrumentalisiert die Kinder und Dritte. Ich bekomme fast täglich Mails von ihm, in denen er mir existentielle Drohungen ausspricht. Ich benötige aufgrund des steten Psychoterror meines Exmannes dringend durchsetzungsstarke, fachkundige Hilfe, wenn möglich von einem Anwalt, der mit so einer Art von Persönlichkeitsstörung vertraut ist.',
    'Beschreiben Sie bitte kurz Ihr Anliegen.: Hallo, im März 2020 fand eine AIDA-Reise von Jamaika aus statt. Der Hinflug wurde von AIDA bei Condor gebucht. Diesbezüglich gibt es für den Kreuzfahrtteilnehmer keine extra Flugbuchung, da diese von AIDA getätigt wurde. Bei Condor wurden jedoch extra Sitzplätze und ein Entertaimentpaket gebucht. Kosten insgesamt 263,94 €. Obwohl noch am Vortag die Bordkarten mit den gebuchten Sitzplätzen von Condor zur Verfügung gestellt wurden, hat man kurzfristig das Fluggerät gewechselt. Weder die gebuchten Sitzplätze noch das Entertaimentpaket konnten genutzt werden. Der Reisemangel wurde bei Condor geltend gemacht. Man verlangt dort die Originalflugbuchung, die natürlich von uns nicht vorgelegt werden kann, da der Flug von AIDA gebucht wurde. Die Bestätigung über die Zusatzleistungen, und nur darum geht es, wurde Condor vorgelegt. Dies interessiert die Condor jedoch nicht. Können Sie hier weiterhelfen? Es besteht eine RS-Versicherung bei der Roland RS-Versicherung.',
    'Guten Tag, Kürzlich ist mein Opa gestorben. Mein Bruder sagte mir das mein Opa ihn zum allein Erben gemacht hat (Haus mit Obstgarten 2000 Quadratmeter + 8000 Quadratmeter Ackerland ). Meinen Vater und seine Stiefschwester hat er enterbt. Mein Vater will sein Pflichtteil einfordern. Zu meiner Frage: steht mir auch etwas zu oder habe ich kein Anspruch ?',
    'Betriebsbedingte Kündigung erhalten nach 5,5 Jahren Betriebszugehörigkeit. Gründe sind für mich nicht nachvollziehbar - Abwicklungsvertrag bietet mir ca. 15.000€ an, was weniger als 3 Monatsgehälter ist. Also Laie habe ich den Eindruck (mein Arbeitgeber hat den Sitz in London und wenig Expertise im Deutschen Arbeitsrecht) das die Abfindung wesentlich höher sein sollte auch weil mein Arbeitgeber unbedingt eine Kündigungsschutzklage vermeiden will.'
]
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
predictions = model.predict(padded)
for pred in predictions:
    print("Legal branch predicted: {} with {}".format(labels[np.argmax(pred)], pred[np.argmax(pred)]))