# Jake Son
# Child Mind Institute

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

p1data = data[(data.participant == 1)].drop(labels=['participant'], axis=1)

p1targets = p1data['target'].values
p1signals = p1data.drop(labels=['target'], axis=1).values

# Output variable (predicted target location) needs to be encoded

encoder = LabelEncoder()
encoder.fit(p1targets)
encoded_Y = encoder.transform(p1targets)
dummy_y = np_utils.to_categorical(encoded_Y)










# Create baseline model for keras multiclass classifier

def baseline_model():

    model = Sequential()
    model.add(Dense(7, input_dim=7, activation='relu'))
    model.add(LSTM(100, return_sequences=True))
    # model.add(LSTM(100))
    model.add(Dense(6, activation='softmax'))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=50, verbose=1)

# Set seed for reliability
seed = 3

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, p1signals, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))






# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
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


