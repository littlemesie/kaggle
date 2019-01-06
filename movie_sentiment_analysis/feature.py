# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/29 21:04
@summary:
"""
import re
import numpy as np
import pandas as pd
import nltk
from gensim.models import Word2Vec

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

def en_word2vec_model():
    """英文模型"""
    path = "../data/word2vec-nlp"
    # 载入数据集
    train, test = get_data()
    """word2vec_model"""
    tokenizer =nltk.data.load('../data/word2vec-nlp/punkt/english.pickle')

    sentences = []
    for i, phrase in enumerate(train['Phrase'].values):
        sentences += phrase_to_sentences(phrase, tokenizer)
    print('预处理 train data...')
    # 模型的构建
    # 模型参数
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    # 训练模型
    print("训练模型中...")
    model = Word2Vec(sentences, workers=num_workers,size=num_features, min_count=min_word_count,
                     window=context, sample=downsampling)
    # 保存模型
    model.init_sims(replace=True)
    model_name = "%s/%s" % (path, "en_word2vec_model")
    model.save(model_name)

def phrase_to_sentences(phrase, tokenizer, remove_stopwords=False):

    raw_sentences = tokenizer.tokenize(phrase)
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:

            sentences.append(phrase_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def phrase_to_wordlist(phrase, remove_stopwords=False):

    phrase_text = re.sub("[^a-zA-Z]"," ", phrase)
    words = phrase_text.lower().split()
    return words

def get_avg_feature_vecs(phrases, model, num_features):
    '''
    给定一个文本列表，每个文本由一个词列表组成，返回每个文本的词向量平均值
    '''
    counter = 0
    phrases_feature_vecs = np.zeros((len(phrases), num_features), dtype="float32")

    for phrase in phrases:
        if counter % 5000 == 0:
            print("Phrase %d of %d" % (counter, len(phrases)))
        feature_vec = make_feature_vec(phrase, model, num_features)

        phrases_feature_vecs[counter] = feature_vec
        counter = counter + 1

    return phrases_feature_vecs

def make_feature_vec(words, model, num_features):
    """对段落中的所有词向量进行取平均操作"""

    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.

    # Index2word包含了词表中的所有词，为了检索速度，保存到set中
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    # 取平均
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def en_word2vec():
    """word2vec"""
    path = "../data/word2vec-nlp"
    model_name = "%s/%s" % (path, "en_word2vec_model")
    model = Word2Vec.load(model_name)
    w = model.wv.doesnt_match("man woman child kitchen".split())
    print(w)
    man_similar = model.wv.most_similar("man", topn=5)
    print(man_similar)

# en_word2vec()
