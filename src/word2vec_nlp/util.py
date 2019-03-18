# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/16 13:48
@summary:
"""
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import nltk
from nltk.corpus import stopwords

path = "../../data/word2vec-nlp"

def get_data():
    """数据预处理"""
    # 载入数据集
    train = pd.read_csv('%s/%s' % (path, 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('%s/%s' % (path, 'testData.tsv'), header=0, delimiter="\t", quoting=3)
    # 预处理数据
    label = train['sentiment']
    train_data = []
    # 每一个句子的单词放在一个列表中
    for i in range(len(train['review'])):
        train_data.append(' '.join(review_to_wordlist(train['review'][i])))
    test_data = []
    for i in range(len(test['review'])):
        test_data.append(' '.join(review_to_wordlist(test['review'][i])))

    return train_data, test_data, label, train, test

def review_to_wordlist(review):
    """从html中解析"""
    soup = BeautifulSoup(review, "html.parser")
    review_text = re.sub('[^a-zA-Z]', ' ',soup.get_text())
    # 转为小写
    lower_case = review_text.lower()
    # 分词
    words = lower_case.split()
    return words

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    '''
    1. 将评论文章，按照句子段落来切分(所以会比文章的数量多很多)
    2. 返回句子列表，每个句子由一堆词组成
    '''
    review = BeautifulSoup(review, "html.parser").get_text()
    # raw_sentences 句子段落集合
    raw_sentences = tokenizer.tokenize(review)
    # print(raw_sentences)

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            # 获取句子中的词列表
            sentences.append(review_to_wordlist_w(raw_sentence, remove_stopwords))
    return sentences

def review_to_wordlist_w(review, remove_stopwords=False):
    """word2vec处理"""
    review_text = re.sub("[^a-zA-Z]"," ", review)

    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # print(words)
    return(words)

def tf_idf(train_data, test_data):
    """TF-IDF向量"""
    tfidf = TFIDF(min_df=2,
                  max_features=None,
                  strip_accents='unicode',
                  analyzer='word',
                  token_pattern=r'\w{1,}',
                  ngram_range=(1, 3),  # 二元文法模型
                  use_idf=1,
                  smooth_idf=1,
                  sublinear_tf=1,
                  stop_words='english')  # 去掉英文停用词

    # 合并训练和测试集以便进行TFIDF向量化操作
    data_all = train_data + test_data
    len_train = len(train_data)

    tfidf.fit(data_all)
    data_all = tfidf.transform(data_all)
    # 恢复成训练集和测试集部分
    train_x = data_all[:len_train]
    test_x = data_all[len_train:]
    print("train: \n", np.shape(train_x[0]))
    print("test: \n", np.shape(test_x[0]))
    return train_x,test_x