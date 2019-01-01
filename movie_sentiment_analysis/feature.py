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
import ngrams
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score

# 获取数据
def get_data():
    train = pd.read_csv("../data/movie/train.tsv", sep='\t')
    test = pd.read_csv("../data/movie/test.tsv", sep='\t')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 4000)
    return train, test


def count(train, test):
    print('Average count of phrases per sentence in train is {0:.0f}.'.format(train.groupby('SentenceId')['Phrase'].count().mean()))
    print('Average count of phrases per sentence in test is {0:.0f}.'.format(test.groupby('SentenceId')['Phrase'].count().mean()))

    print('Number of phrases in train: {}. Number of sentences in train: {}.'.format(train.shape[0], len(train.SentenceId.unique())))
    print('Number of phrases in test: {}. Number of sentences in test: {}.'.format(test.shape[0], len(test.SentenceId.unique())))

    print('Average word length of phrases in train is {0:.0f}.'.format(np.mean(train['Phrase'].apply(lambda x: len(x.split())))))
    print('Average word length of phrases in test is {0:.0f}.'.format(np.mean(test['Phrase'].apply(lambda x: len(x.split())))))

def text(train, test):
    text = ' '.join(train.loc[train.Sentiment == 4, 'Phrase'].values)
    # text_trigrams = [i for i in ngrams(text.split(), 3)

def logistic(train, test):
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


if __name__ == '__main__':
    train, test = get_data()
    logistic(train, test)