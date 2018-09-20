# Jake Son
# Child Mind Institute

import keras
import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# BINARY CLASSIFICATION TUTORIAL

# fix random seed for reproducibility
np.random.seed(7)

# Load data but only keep the top n words, zero the rest
top_words = 500
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=50)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# TINGLE ANALYSIS

df = pd.read_csv('tingle_pilot_data_shareable.csv')

data = df[['distance',
           'pitch',
           'roll',
           'target',
           'thermopile1',
           'thermopile2',
           'thermopile3',
           'thermopile4',
           'ontarget',
           'participant']]

data = data[(data.target == 'rotate-mouth') |
            (data.target == 'rotate-nose') |
            (data.target == 'rotate-cheek') |
            (data.target == 'rotate-eyebrow') |
            (data.target == 'rotate-top-head') |
            (data.target == 'rotate-back-head')]

data = data[(data.ontarget == True)]

data = data.drop(labels=['ontarget'], axis=1)

p1data = data[(data.participant == 2)].drop(labels=['participant'], axis=1)

p1targets = p1data['target'].values
p1signals = p1data.drop(labels=['target'], axis=1).values

# Output variable (predicted target location) needs to be encoded

encoder = LabelEncoder()
encoder.fit(p1targets)
encoded_Y = encoder.transform(p1targets)
dummy_y = np_utils.to_categorical(encoded_Y)


# Create binary classification model

df = pd.read_csv('tingle_pilot_data_shareable.csv')

data = df[['distance',
           'pitch',
           'roll',
           'target',
           'thermopile1',
           'thermopile2',
           'thermopile3',
           'thermopile4',
           'ontarget',
           'participant']]

data = data[(data.target == 'rotate-mouth') |
            (data.target == 'rotate-nose') |
            (data.target == 'rotate-cheek') |
            (data.target == 'rotate-eyebrow') |
            (data.target == 'rotate-top-head') |
            (data.target == 'rotate-back-head')]

data = data[(data.ontarget == True)]

data = data.drop(labels=['ontarget'], axis=1)

p1data = data[(data.participant == 1)].drop(labels=['participant'], axis=1)

p1targets = list(p1data['target'].values)
p1targets = np.array([1 if x == 'rotate-back-head' else 0 for x in p1targets])
p1signals = p1data.drop(labels=['target'], axis=1).values

x_train, x_test, y_train, y_test = train_test_split(p1signals, p1targets, test_size=.33)

model = Sequential()
model.add(Dense(60, input_dim=7, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32, verbose=1)

# def create_baseline():
#     # create model
#     model = Sequential()
#     model.add(Dense(60, input_dim=7, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline, epochs=10, batch_size=32, verbose=1)
# kfold = StratifiedKFold(n_splits=2, shuffle=True)
# results = cross_val_score(estimator, p1signals, p1targets, cv=kfold)
# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
