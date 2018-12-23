# -*- coding: utf-8 -*-

import jieba
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # jieba custom setting.
    jieba.set_dictionary('../data/word2vec-nlp/jieba_dict/dict.txt.big')

    # load stopwords set
    stopword_set = set()
    with open('../data/word2vec-nlp/jieba_dict/stopwords.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))

    output = open('../data/word2vec-nlp/wiki_seg.txt', 'w', encoding='utf-8')
    with open('../data/word2vec-nlp/wiki_zh.txt', 'r', encoding='utf-8') as content :
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopword_set:
                    output.write(word + ' ')
            output.write('\n')

            if (texts_num + 1) % 10000 == 0:
                logging.info("已完成前 %d 行的分词" % (texts_num + 1))
    output.close()

if __name__ == '__main__':
    main()