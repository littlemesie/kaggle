# -*- coding: utf-8 -*-

import logging
import sys

from gensim.corpora import WikiCorpus


def main():
    """下载页面https://dumps.wikimedia.org/zhwiki/"""
    # if len(sys.argv) != 2:
    #     print("Usage: python3 " + sys.argv[0] + " wiki_data_path")
    #     exit()
    #
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # wiki_corpus = WikiCorpus(sys.argv[1], dictionary={})
    path = "../data/word2vec-nlp/zhwiki-20181201-pages-articles.xml.bz2"
    wiki_corpus = WikiCorpus(path, dictionary={})
    texts_num = 0

    with open("../data/word2vec-nlp/wiki_texts.txt",'w',encoding='utf-8') as output:
        for text in wiki_corpus.get_texts():
            output.write(' '.join(text) + '\n')
            texts_num += 1
            if texts_num % 10000 == 0:
                logging.info("已处理 %d 篇文章" % texts_num)

if __name__ == "__main__":
    main()
    " opencc -i wiki_texts.txt -o wiki_zh.txt -c t2s.json"