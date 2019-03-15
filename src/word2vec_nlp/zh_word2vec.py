# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/16 14:28
@summary:
"""

import logging
from gensim.models import word2vec


def zh_word2vec():
    path = "../data/word2vec-nlp"
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(path + "/wiki_seg.txt")
    model = word2vec.Word2Vec(sentences, size=250)

    #保存模型
    model.save(path + "/word2vec.model")

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec.load("../data/word2vec-nlp/word2vec.model")
    res = model.wv.similarity('中国','日本')
    print(res)
    #
    # while True:
    #     try:
    #         query = input()
    #         q_list = query.split()
    #
    #         if len(q_list) == 1:
    #             print("相似詞前 100 排序")
    #             res = model.most_similar(q_list[0], topn=100)
    #             for item in res:
    #                 print(item[0] + "," + str(item[1]))
    #
    #         elif len(q_list) == 2:
    #             print("計算 Cosine 相似度")
    #             res = model.similarity(q_list[0], q_list[1])
    #             print(res)
    #         else:
    #             print("%s之於%s，如%s之於" % (q_list[0], q_list[2], q_list[1]))
    #             res = model.most_similar([q_list[0], q_list[1]], [q_list[2]], topn=100)
    #             for item in res:
    #                 print(item[0] + "," + str(item[1]))
    #         print("----------------------------")
    #     except Exception as e:
    #         print(repr(e))

if __name__ == "__main__":
    # zh_word2vec()
    main()