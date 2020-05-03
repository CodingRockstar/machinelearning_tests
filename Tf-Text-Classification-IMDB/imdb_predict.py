# IMDB sentiment analysis from tutorial with own prediction block
# https://www.youtube.com/watch?v=Rc2XHfk_jss
from __future__ import absolute_import, division, print_function
import pathlib

import tensorflow as tf
from tensorflow import keras

import numpy as np

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()

word_index = {k: (v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# load model
model = tf.keras.models.load_model(str(pathlib.Path(__file__).parent.absolute()) + "/imdb_model.h5")
# summarize model.
# model.summary()


# make predictions
def encode_review(text):
    return [word_index.get(i, word_index.get("<UNK>")) for i in text.split(" ")]

reviews = [
    "This film was really awesome. Cant wait to see it again!",
    "What a terrible movie. Hate it! Never again!",
    "lovely film. Wonderful characters with a lovely story!"
]
for review in reviews:
    Xnew = [[0] + encode_review(review)]
    ynew = model.predict_classes(Xnew)
    # show the inputs and predicted outputs
    print("X=%s, Predicted=%s" % (review, ynew[0]))