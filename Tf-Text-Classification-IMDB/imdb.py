# IMDB sentiment analysis from tutorial with own prediction block
# https://www.youtube.com/watch?v=Rc2XHfk_jss
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#print(f"Training entries {len(train_data)}. Labels: {len(train_labels)}")

#print(len(train_data[0]), len(train_data[1]))

word_index = imdb.get_word_index()

word_index = {k: (v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, "?") for i in text])

# print(decode_review(train_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data, 
                                                        value=word_index["<PAD>"],
                                                        padding="post",
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, 
                                                       value=word_index["<PAD>"],
                                                       padding="post",
                                                       maxlen=256)

vocab_size = 10000

# create neural network
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))           # input layer
model.add(keras.layers.GlobalAveragePooling1D())            # 1st hidden layer
model.add(keras.layers.Dense(16, activation=tf.nn.relu))    # 2nd hidden layer /w 16 hidden units
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))  # output layer (single output node --> 0 or 1)

#model.summary()
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["acc"])

# add validation sets
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=30,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# output accuracy
results = model.evaluate(test_data, test_labels)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))


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