# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
import gensim
import nltk
from nltk.corpus import stopwords

def get_data():
    """数据预处理"""
    path = "../data/word2vec-nlp"
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

    return train_data, test_data, label, test

def review_to_wordlist(review):
    soup = BeautifulSoup(review, "html.parser")
    review_text = re.sub('[^a-zA-Z]', ' ',soup.get_text())
    # 转为小写
    lower_case = review_text.lower()
    # 分词
    words = lower_case.split()
    return words

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

def logistic(train_x, test_x, test,label):
    # 设定grid search的参数
    grid_values = {'C': [1, 15, 30, 50]}
    # grid_values = {'C': [30]}
    # 设定打分为roc_auc
    """
    penalty: l1 or l2, 用于指定惩罚中使用的标准。
    """
    model_LR = GridSearchCV(LR(penalty='l2', dual=True, random_state=0), grid_values, scoring='roc_auc', cv=20)
    model_LR.fit(train_x, label)
    print(model_LR.cv_results_, '\n', model_LR.best_params_, model_LR.best_score_)
    model_LR = LR(penalty='l2', dual=True, random_state=0)
    model_LR.fit(train_x, label)

    test_predicted = np.array(model_LR.predict(test_x))
    print("结果:")
    submission_df = pd.DataFrame(data={'id': test['id'], 'sentiment': test_predicted})
    print(submission_df.head(10))

def word2vec():
    """word2vec"""
if __name__ == '__main__':
    train_data, test_data, label , test = get_data()
    train_x, test_x = tf_idf(train_data, test_data)
    # naive_bayes(train_x, test_x, test,label)
    logistic(train_x, test_x, test, label)


