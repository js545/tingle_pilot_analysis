# Jake Son
# Child Mind Institute

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data and remove columns that are not in use

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

# Subset data collected from target locations of interest

data = data[(data.target == 'rotate-mouth') |
            (data.target == 'rotate-nose') |
            (data.target == 'rotate-cheek') |
            (data.target == 'rotate-eyebrow') |
            (data.target == 'rotate-top-head') |
            (data.target == 'rotate-back-head') |
            (data.target == 'offbody-ceiling') |
            (data.target == 'offbody-floor') |
            (data.target == 'offbody-+') |
            (data.target == 'offbody-X') |
            (data.target == 'offbody-spiral')]

data = data[(data.ontarget == True)]

data = data.drop(labels=['ontarget'], axis=1)

# Subset data from a given participant

# for part_num in range(1, len(data.participant.unique()) + 1):
for part_num in range(1, 3):

    # part_num = 1

    p1data = data[(data.participant == part_num)].drop(labels=['participant'], axis=1)

    # Select on-target location

    for target_location in ['rotate-mouth', 'rotate-nose', 'rotate-cheek', 'rotate-eyebrow', 'rotate-top-head',
                            'rotate-back-head']:

        # target_location = 'rotate-mouth'

        p1targets = list(p1data['target'].values)
        p1targets = np.array([1 if x == target_location else 0 for x in p1targets])
        p1signals = p1data.drop(labels=['target'], axis=1).values

        # Separate into train/test sets

        x_train, x_test, y_train, y_test = train_test_split(p1signals, p1targets, test_size=.33)

        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

        # Create LSTM model

        model = Sequential()
        model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=1, shuffle=False)
        # history = model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1, shuffle=False)

        sig = model.predict_classes(x_test)

        # Plot loss and accuracy
        # plt.figure()
        # plt.title(str('Participant {} target {}').format(part_num, target_location))
        # plt.plot(history.history['loss'], label='train')
        # plt.plot(history.history['val_loss'], label='test')
        # plt.legend()
        # plt.show()

        plt.figure()
        plt.title(str('Participant {} target {}').format(part_num, target_location))
        plt.ylim([.5, 1])
        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='test')
        plt.legend()
        plt.show()