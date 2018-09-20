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
import matplotlib.pyplot as plt

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

p1data = data[(data.participant == 5)].drop(labels=['participant'], axis=1)

p1targets = list(p1data['target'].values)
p1targets = np.array([1 if x == 'rotate-nose' else 0 for x in p1targets])
p1signals = p1data.drop(labels=['target'], axis=1).values

x_train, x_test, y_train, y_test = train_test_split(p1signals, p1targets, test_size=.33)

model = Sequential()
model.add(Dense(60, input_dim=7, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32, verbose=1)

# Visualize training / validation accuracy

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


for num in range(len(data.participant.unique())+1):

    pdata = data[(data.participant == num)].drop(labels=['participant'], axis=1)
    ptargets = list(pdata['target'].values)
    psignals = pdata.drop(labels=['target'], axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(psignals, ptargets, test_size=.33)

    model = Sequential()
    model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.compile(loss='binary_crossentropy', optimiizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32, verbose=0)

    history.history['acc']
    history.history['loss']


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
