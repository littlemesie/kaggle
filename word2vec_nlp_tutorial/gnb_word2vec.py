# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/16 15:14
@summary: 高斯贝叶斯+Word2vec训练
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.cross_validation import cross_val_score
from gensim.models import Word2Vec
from word2vec_nlp_tutorial.util import get_data
from word2vec_nlp_tutorial import word2vec_model

def gbn_word2vec():
    """"""
    model_GNB = GNB()
    train_data, test_data, label, train, test = get_data()
    path = "../data/word2vec-nlp"
    model_name = "%s/%s" % (path, "300features_40minwords_10context")
    model = Word2Vec.load(model_name)
    train_data_vecs = word2vec_model.get_avg_feature_vecs(train_data,model, 300)
    test_data_vecs = word2vec_model.get_avg_feature_vecs(test_data, model, 300)
    model_GNB.fit(train_data_vecs, label)

    print("高斯贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_GNB, train_data_vecs, label, cv=10, scoring='roc_auc')))

    print('保存结果...')
    result = model_GNB.predict(test_data_vecs)
    submission_df = pd.DataFrame(data={'id': test['id'], 'sentiment': result})
    print(submission_df.head(10))

gbn_word2vec()