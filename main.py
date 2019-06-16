
import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.models import load_model
from keras import initializers, regularizers, constraints, optimizers, layers


train = pd.read_csv("train.csv", index_col='id')
test = pd.read_csv("test.csv", index_col='id')
#test_dataset = pd.read_csv("test.csv")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

inp = Input(shape=(maxlen, ))
embed_size = 128
x = Embedding(max_features, embed_size)(inp)

x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

#model = load_model('partly_trained.h5')
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
batch_size = 128
epochs = 1
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model.save('partly_trained.h5')

res = model.predict(X_te, verbose=1)
arr = np.asarray(test.index)
df = pd.DataFrame(res, index = arr, columns=["toxic", "severe_toxic","obscene","threat","insult","identity_hate"])
df.to_csv("output.csv")