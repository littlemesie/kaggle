# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/16 14:41
@summary:
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from word2vec_nlp import util

def logistic(train_x, test_x, test, label):

    model_LR = LR(penalty='l2', dual=True, random_state=0)
    model_LR.fit(train_x, label)
    print("准确率：%s" % model_LR.score(train_x,label))
    test_predicted = np.array(model_LR.predict(test_x))
    print("结果:")
    submission_df = pd.DataFrame(data={'id': test['id'], 'sentiment': test_predicted})
    print(submission_df.head(10))

if __name__ == '__main__':
    train_data, test_data, label, train, test = util.get_data()
    train_x, test_x = util.tf_idf(train_data, test_data)
    logistic(train_x, test_x, test, label)