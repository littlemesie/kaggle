# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from titanic import util


train, test = util.read_csv()
data_trainX, data_testX, data_trainY, data_testY = util.train_data(train)

def test_acc(y_pred, data_testY):
    acc_sum = 0
    data_testY = np.array(data_testY)
    for i, y in enumerate(y_pred):
        if y == data_testY[i]:
            acc_sum += 1
    acc = acc_sum / len(y_pred)
    return acc

logreg = LogisticRegression()
logreg.fit(data_trainX, data_trainY)
log_y_pred = logreg.predict(data_testX)
train_acc_log = round(logreg.score(data_trainX, data_trainY), 4)
test_acc_log = test_acc(log_y_pred, data_testY)
print(train_acc_log)
print(test_acc_log)

gaussian = GaussianNB()
gaussian.fit(data_trainX, data_trainY)
nb_y_pred = gaussian.predict(data_testX)
train_acc_gaussian = round(gaussian.score(data_trainX, data_trainY), 4)
test_acc_gaussian = test_acc(nb_y_pred, data_testY)
print(train_acc_gaussian)
print(test_acc_gaussian)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(data_trainX, data_trainY)
rf_y_pred = random_forest.predict(data_testX)
train_acc_random_forest = round(random_forest.score(data_trainX, data_trainY), 4)
test_acc_random_forest = test_acc(nb_y_pred, data_testY)
print(train_acc_random_forest)
print(test_acc_random_forest)