# Jake Son
# Child Mind Institute

import os
import numpy as np
import pandas as pd
from sklearn import metrics
from keras.layers import LSTM
from statsmodels import robust
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('tingle_pilot_data_shareable.csv')

data = df[['distance', 'pitch', 'roll', 'target',
           'thermopile1', 'thermopile2', 'thermopile3', 'thermopile4',
           'ontarget', 'participant']]

# data = df[['distance', 'pitch', 'roll', 'target', 'ontarget', 'participant']]

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

cm_results = {'rotate-mouth': [],
              'rotate-nose': [],
              'rotate-cheek': [],
              'rotate-eyebrow': [],
              'rotate-top-head': [],
              'rotate-back-head': []}

for part_num in range(1, len(data.participant.unique()) + 1):
# for part_num in range(1, 3):

    if part_num != 22:

        p1data = data[(data.participant == part_num)].drop(labels=['participant'], axis=1)

        # col_subset = ['distance', 'pitch', 'roll']
        col_subset = ['distance', 'pitch', 'roll', 'thermopile1', 'thermopile2', 'thermopile3', 'thermopile4']

        p1data[col_subset] = StandardScaler().fit_transform(p1data[col_subset])

        for target_loc in targets:

            print(str('Analyzing participant {} target {}').format(part_num, target_loc))

            p1targets = list(p1data['target'].values)
            p1targets = np.array([1 if x == target_loc else 0 for x in p1targets])
            p1signals = p1data.drop(labels=['target'], axis=1).values

            x_train, x_test, y_train, y_test = train_test_split(p1signals, p1targets, test_size=.25, shuffle=True, random_state=1)

            x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
            x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

            model = Sequential()
            model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
            model.add(LSTM(50, dropout=.2))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_test, y_test), verbose=0, shuffle=False)

            pred = model.predict_classes(x_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)

            report = metrics.classification_report(y_test, pred, output_dict=True)
            precision = round(report['weighted avg']['precision'], 2)
            recall = round(report['weighted avg']['recall'], 2)
            f1score = round(report['weighted avg']['f1-score'], 2)
            auroc = (metrics.auc(fpr, tpr))

            # print((metrics.auc(fpr, tpr), f1score))

            cm = confusion_matrix(y_test, pred)

            tn, fp, fn, tp = cm.ravel()

            cm_results[target_loc].append(cm)

            results_dict[str('{}_{}').format(part_num, target_loc)] = [part_num, target_loc, precision,
                                                                       recall, f1score, auroc]

    else:

        continue

df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['participant', 'target', 'precision', 'recall', 'f1score', 'auroc'])
df = df.set_index('participant')

# df.to_csv('~/Documents/CMI/tingle_pilot_analysis/group_lstm_analysis.csv')

for num in range(6):

    cm_results[targets[num]] = np.stack(cm_results[targets[num]])
    np.save(str('/Users/jakeson/Documents/CMI/tingle_pilot_analysis/confusion_matrices/{}').format(targets[num]), cm_results[targets[num]])

#######################################################################################################################
# Confusion matrix analysis

# Normalization of a confusion matrix named cm

import itertools
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, weight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, weight='bold')
    plt.yticks(tick_marks, classes, weight='bold')

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', weight='bold')
    plt.xlabel('Predicted label', weight='bold')
    plt.tight_layout()

for target in targets:

    print(target)

    cm = np.load(str('/Users/jakeson/Documents/CMI/tingle_pilot_analysis/confusion_matrices/{}.npy').format(target))
    cm = np.ndarray.astype(cm, 'float')

    for num in range(cm.shape[0]):

        cm[num] = cm[num].astype('float') / cm[num].sum(axis=1)[:, np.newaxis]

    avg_cm = np.average(cm, axis=0)

    print(avg_cm)

    plot_confusion_matrix(avg_cm, classes=['off-target', 'on-target'], normalize=False, title=str(target).replace('rotate-', ''), cmap=plt.cm.Blues)
    plt.savefig(str('/Users/jakeson/Documents/CMI/tingle_pilot_analysis/confusion_matrices/{}.png').format(target),
                dpi=300)
    plt.show()




# Analysis using simple paired t-test of AUROC values with and without thermal information

from scipy.stats import ttest_rel

df_n = pd.read_csv('auroc_n_thermal.csv')
df_y = pd.read_csv('auroc_y_thermal.csv')

for target in df_n.target.unique():

    df_n_sub = df_n[df_n.target == target]
    df_y_sub = df_y[df_y.target == target]

    output = ttest_rel(df_n_sub.auroc.tolist(), df_y_sub.auroc.tolist())
    stats = output[0] # negative stats value means values from second input > first input
    p_val = output[1]

    print((target, stats, p_val))


# ## Group level analysis
#
# os.chdir('/Users/jakeson/Documents/CMI/tingle_pilot_analysis')
#
# df = pd.read_csv('tingle_pilot_data_shareable.csv')
#
#
# data = df[['distance', 'pitch', 'roll', 'target',
#            'thermopile1', 'thermopile2', 'thermopile3', 'thermopile4',
#            'ontarget', 'participant']]
#
# data = data[(data.target == 'rotate-mouth') | (data.target == 'rotate-nose') | (data.target == 'rotate-cheek') |
#             (data.target == 'rotate-eyebrow') | (data.target == 'rotate-top-head') |
#             (data.target == 'rotate-back-head') | (data.target == 'offbody-ceiling') |
#             (data.target == 'offbody-floor') | (data.target == 'offbody-+') |
#             (data.target == 'offbody-X') | (data.target == 'offbody-spiral')]
#
# data = data[(data.ontarget == True)]
#
# data = data.drop(labels=['ontarget'], axis=1)
#
# targets = ['rotate-mouth', 'rotate-nose', 'rotate-cheek', 'rotate-eyebrow', 'rotate-top-head',
#                             'rotate-back-head']
#
# results_dict = {}
#
# print('Beginning analysis')
#
# for part_num in range(1, len(data.participant.unique()) + 1):
#
#     if part_num != 22:
#
#         train_data = data[(data.participant != part_num)].drop(labels=['participant'], axis=1)
#         test_data = data[(data.participant == part_num)].drop(labels=['participant'], axis=1)
#
#         col_subset = ['distance', 'pitch', 'roll', 'thermopile1', 'thermopile2', 'thermopile3', 'thermopile4']
#
#         train_data[col_subset] = StandardScaler().fit_transform(train_data[col_subset])
#         test_data[col_subset] = StandardScaler().fit_transform(test_data[col_subset])
#
#         for target_loc in targets:
#
#             print(str('Analyzing participant {} target {}').format(part_num, target_loc))
#
#             train_targets = list(train_data['target'].values)
#             train_targets = np.array([1 if x == target_loc else 0 for x in train_targets])
#             train_signals = train_data.drop(labels=['target'], axis=1).values
#
#             x_train, x_test0, y_train, y_test0 = train_test_split(train_signals, train_targets, test_size=0, shuffle=True, random_state=1)
#             x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
#
#             test_targets = list(test_data['target'].values)
#             test_targets = np.array([1 if x == target_loc else 0 for x in test_targets])
#             test_signals = test_data.drop(labels=['target'], axis=1).values
#
#             x_trainz, x_test, y_trainz, y_test = train_test_split(test_signals, test_targets, test_size=0, shuffle=True, random_state=1)
#             x_trainz = x_trainz.reshape((x_trainz.shape[0], 1, x_trainz.shape[1]))
#
#             model = Sequential()
#             model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
#             model.add(LSTM(50, dropout=.2))
#             model.add(Dense(1, activation='sigmoid'))
#             model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#             history = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_trainz, y_trainz), verbose=0, shuffle=False)
#
#             pred = model.predict_classes(x_trainz)
#             pred_proba = model.predict_proba(x_trainz)
#             fpr, tpr, thresholds = metrics.roc_curve(y_trainz, pred_proba)
#
#             report = metrics.classification_report(y_trainz, pred, output_dict=True)
#             precision = round(report['weighted avg']['precision'], 2)
#             recall = round(report['weighted avg']['recall'], 2)
#             f1score = round(report['weighted avg']['f1-score'], 2)
#             auroc = (metrics.auc(fpr, tpr))
#
#             print((auroc, f1score))
#
#             results_dict[str('{}_{}').format(part_num, target_loc)] = [part_num, target_loc, precision, recall, f1score, auroc]
#
#     else:
#
#         continue
#
# df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['participant', 'target', 'precision', 'recall', 'f1score', 'auroc'])
# df = df.set_index('participant')
#
# df.to_csv('group_level_analysis_fixed.csv')
#
# df = pd.read_csv('group_level_analysis_fixed.csv')
#
# for col in df.target.unique():
#
#     subset = df[df.target == col]
#
#     med = np.median(subset.auroc.tolist())
#     mad = robust.mad(subset.auroc.tolist())
#
#     print(col, med, mad)







