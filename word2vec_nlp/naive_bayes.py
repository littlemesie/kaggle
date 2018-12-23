# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/16 14:36
@summary:
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import cross_val_score

def naive_bayes(train_x, test_x,test,label):
    """朴素贝叶斯"""
    model_NB = MNB()  # (alpha=1.0, class_prior=None, fit_prior=True)
    # 为了在预测的时候使用
    model_NB.fit(train_x, label)

    print("多项式贝叶斯分类器10折交叉验证得分:  \n", cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc'))
    print("多项式贝叶斯分类器10折交叉验证平均得分: ", np.mean(cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc')))

    test_predicted = np.array(model_NB.predict(test_x))
    submission_df = pd.DataFrame(data={'id': test['id'], 'sentiment': test_predicted})
    print("结果:")
    print(submission_df.head(100))