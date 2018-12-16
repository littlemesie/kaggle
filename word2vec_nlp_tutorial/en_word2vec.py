# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/16 14:29
@summary:
"""
from gensim.models import Word2Vec
from word2vec_nlp_tutorial.word2vec_model import get_avg_feature_vecs
from word2vec_nlp_tutorial.util import review_to_wordlist_w
path = "../data/word2vec-nlp"

def en_word2vec():
    """word2vec"""
    model_name = "%s/%s" % (path, "300features_40minwords_10context")
    model = Word2Vec.load(model_name)
    w = model.wv.doesnt_match("man woman child kitchen".split())
    print(w)
    man_similar = model.wv.most_similar("man", topn=5)
    print(man_similar)
    si = model.wv.most_similar(positive=['woman', 'king'], topn=1)
    print(si)
    test_data = "Watching Time Chasers, it obvious that it was made by a bunch of friends"
    test_data = review_to_wordlist_w(test_data)
    test_data_vecs = get_avg_feature_vecs(test_data, model, 300)
    print(test_data_vecs)

en_word2vec()