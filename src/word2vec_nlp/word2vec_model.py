# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/16 14:42
@summary:
"""
import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec
from word2vec_nlp.util import review_to_sentences

def en_word2vec_model():
    """英文模型"""
    path = "../data/word2vec-nlp"
    # 载入数据集
    train = pd.read_csv('%s/%s' % (path, 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    """word2vec_model"""
    tokenizer =nltk.data.load('../data/word2vec-nlp/punkt/english.pickle')
    print(tokenizer)
    sentences = []
    for i, review in enumerate(train["review"]):
        # print(i, review)
        sentences += review_to_sentences(review, tokenizer)
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
    model_name = "%s/%s" % (path, "300features_40minwords_10context")
    model.save(model_name)

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


def get_avg_feature_vecs(reviews, model, num_features):
    '''
    给定一个文本列表，每个文本由一个词列表组成，返回每个文本的词向量平均值
    '''
    counter = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        if counter % 5000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))

        review_feature_vecs[counter] = make_feature_vec(review, model, num_features)
        counter = counter + 1

    return review_feature_vecs