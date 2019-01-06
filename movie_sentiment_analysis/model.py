# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/29 21:04
@summary:
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB as MNB
from gensim.models import Word2Vec

from movie_sentiment_analysis import feature

def logistic(train, test):
    """逻辑回归"""
    train_x, test_x = tf_idf(train, test)
    y = train['Sentiment']
    logreg = LR()
    ovr = OneVsRestClassifier(logreg)

    ovr.fit(train_x, y)
    scores = cross_val_score(ovr, train_x, y, scoring='accuracy', n_jobs=-1, cv=3)
    print('Cross-validation mean accuracy {0:.2f}%'.format(np.mean(scores) * 100))

def naive_bayes(train, test):
    """贝叶斯"""
    train_x, test_x = tf_idf(train, test)
    y = train['Sentiment']
    model_NB = MNB()
    scores = np.mean(cross_val_score(model_NB, train_x, y, cv=3, n_jobs=2, scoring='accuracy'))
    print('Cross-validation mean accuracy {0:.2f}%'.format(scores * 100))


def random_forest_word2vec(train, test):
    """随机森林+word2vec"""
    train_x = train['Phrase']
    test_x = test['Phrase']

    y = train['Sentiment']
    path = "../data/word2vec-nlp"
    model_name = "%s/%s" % (path, "en_word2vec_model")
    model = Word2Vec.load(model_name)
    train_data_vecs = feature.get_avg_feature_vecs(train_x, model, 300)
    train_data_vecs[np.isnan(train_data_vecs)] = np.mean(train_data_vecs[~np.isnan(train_data_vecs)])

    test_data_vecs = feature.get_avg_feature_vecs(test_x, model, 300)
    test_data_vecs[np.isnan(test_data_vecs)] = np.mean(test_data_vecs[~np.isnan(test_data_vecs)])
    forest = RandomForestClassifier(n_estimators=30, n_jobs=2)
    forest = forest.fit(train_data_vecs, y)
    scores = np.mean(cross_val_score(forest, train_data_vecs, y, cv=3, n_jobs=-1, scoring='accuracy'))
    print("Cross-validation mean accuracy {0:.2f}% ".format(scores * 100))

    # 测试集
    # result = forest.predict(test_data_vecs)

def tf_idf(train_data, test_data):
    len_train = len(train_data)
    tokenizer = TweetTokenizer()
    vectorizer = TFIDF(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)
    full_text = list(train_data['Phrase'].values) + list(test_data['Phrase'].values)
    vectorizer.fit(full_text)
    data_all = vectorizer.transform(full_text)
    # 恢复成训练集和测试集部分
    train_x = data_all[:len_train]
    test_x = data_all[len_train:]
    return train_x, test_x

if __name__ == '__main__':
    train, test = feature.get_data()
    # logistic(train, test)
    # naive_bayes(train, test)
    random_forest_word2vec(train, test)

