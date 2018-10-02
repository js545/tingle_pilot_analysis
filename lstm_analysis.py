# Jake Son
# Child Mind Institute

import numpy as np
import pandas as pd
from sklearn import metrics
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

df = pd.read_csv('tingle_pilot_data_shareable.csv')

data = df[['distance', 'pitch', 'roll', 'target',
           'thermopile1', 'thermopile2', 'thermopile3', 'thermopile4',
           'ontarget', 'participant']]

data = data[(data.target == 'rotate-mouth') | (data.target == 'rotate-nose') | (data.target == 'rotate-cheek') |
            (data.target == 'rotate-eyebrow') | (data.target == 'rotate-top-head') |
            (data.target == 'rotate-back-head') | (data.target == 'offbody-ceiling') |
            (data.target == 'offbody-floor') | (data.target == 'offbody-+') |
            (data.target == 'offbody-X') | (data.target == 'offbody-spiral')]

data = data[(data.ontarget == True)]

data = data.drop(labels=['ontarget'], axis=1)

targets = ['rotate-mouth', 'rotate-nose', 'rotate-cheek', 'rotate-eyebrow', 'rotate-top-head',
                            'rotate-back-head']

results_dict = {}

for part_num in range(23, len(data.participant.unique()) + 1):

    p1data = data[(data.participant == part_num)].drop(labels=['participant'], axis=1)

    for target_loc in targets:

        print(str('Analyzing participant {} target {}').format(part_num, target_loc))

        p1targets = list(p1data['target'].values)
        p1targets = np.array([1 if x == target_loc else 0 for x in p1targets])
        p1signals = p1data.drop(labels=['target'], axis=1).values

        x_train, x_test, y_train, y_test = train_test_split(p1signals, p1targets, test_size=.25, shuffle=True)

        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

        model = Sequential()
        model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
        model.add(LSTM(50, dropout=.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=0, shuffle=False)

        sig = model.predict_classes(x_test)

        report = metrics.classification_report(y_test, sig, output_dict=True)
        precision = round(report['weighted avg']['precision'], 2)
        recall = round(report['weighted avg']['recall'], 2)
        f1score = round(report['weighted avg']['f1-score'], 2)

        results_dict[str('{}_{}').format(part_num, target_loc)] = [part_num, target_loc, precision, recall, f1score]

df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['participant', 'target', 'precision', 'recall', 'f1score'])
df = df.set_index('participant')

df.to_csv('~/Documents/CMI/tingle_pilot_analysis/results.csv')