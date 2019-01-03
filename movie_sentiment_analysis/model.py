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

from movie_sentiment_analysis import feature

def logistic(train, test):
    """逻辑回归"""
    tokenizer = TweetTokenizer()
    vectorizer = TFIDF(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)
    full_text = list(train['Phrase'].values) + list(test['Phrase'].values)
    vectorizer.fit(full_text)
    train_vectorized = vectorizer.transform(train['Phrase'])
    test_vectorized = vectorizer.transform(test['Phrase'])
    y = train['Sentiment']
    logreg = LR()
    ovr = OneVsRestClassifier(logreg)

    ovr.fit(train_vectorized, y)
    scores = cross_val_score(ovr, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)
    print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))


def random_forest(train, test):
    """随机森林"""
    train_x, test_x = tf_idf(train, test)
    forest = RandomForestClassifier(n_estimators=100, n_jobs=2)
    label = train['Sentiment']
    forest = forest.fit(train_x, label)
    print('Score: ', forest.score(train_x, label))
    # 测试集
    result = forest.predict(test_x)

    print('保存结果...')
    submission_df = pd.DataFrame(data={'PhraseId': test['PhraseId'], 'Sentiment': result})
    print(submission_df.head(10))

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
    random_forest(train, test)