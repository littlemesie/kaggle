# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/16 15:33
@summary: 随机森林+Word2vec训练
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from gensim.models import Word2Vec
from word2vec_nlp.util import get_data
from word2vec_nlp import word2vec_model
from sklearn.ensemble import RandomForestClassifier

def random_forest_word2vec():
    train_data, test_data, label, train, test = get_data()
    path = "../data/word2vec-nlp"
    model_name = "%s/%s" % (path, "300features_40minwords_10context")
    model = Word2Vec.load(model_name)
    train_data_vecs = word2vec_model.get_avg_feature_vecs(train_data, model, 300)
    test_data_vecs = word2vec_model.get_avg_feature_vecs(test_data, model, 300)
    forest = RandomForestClassifier(n_estimators=100, n_jobs=2)
    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(train_data_vecs, label)
    print("train_data_vecs: ", np.mean(cross_val_score(forest, train_data_vecs, label, cv=10, scoring='roc_auc')))

    # 测试集
    result = forest.predict(test_data_vecs)

    print('保存结果...')
    submission_df = pd.DataFrame(data={'id': test['id'], 'sentiment': result})
    print(submission_df.head(10))