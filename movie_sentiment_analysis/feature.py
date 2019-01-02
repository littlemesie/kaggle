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
