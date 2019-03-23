# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/3/23 16:46
@summary:
"""
import jieba,os,json

def get_label(lable):
    if lable == "自动摘要":
        return "__label__0"

    if lable == "机器翻译":
        return "__label__1"

    if lable == "机器作者":
        return "__label__2"
    if lable == "人类作者":
        return "__label__3"

def get_train_data():
    ftrain = open("../../../data/smp/news_train.txt","w")
    with open("../../../data/smp/training.txt","r") as f:
        for line in f.readlines():
            data = json.loads(line)
            label = get_label(data['标签'])
            seg_text = jieba.cut(data['内容'].replace("\t", " ").replace("\n", " "))
            outline = " ".join(seg_text)
            outline = outline + "\t" + label + "\n"
            ftrain.write(outline)
            ftrain.flush()
    ftrain.close()

def get_test_data():
    ftest = open("../../../data/smp/news_test.txt", "w")
    with open("../../../data/smp/testing.txt", "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            seg_text = jieba.cut(data['内容'].replace("\t", " ").replace("\n", " "))
            outline = " ".join(seg_text) + "\n"
            ftest.write(outline)
            ftest.flush()
            print(data)
            # quit()

# get_train_data()
# get_test_data()

def load_data():
    ftrain = open("../../data/smp/train.txt", "w", encoding='utf-8')
    ftest = open("../../data/smp/test.txt", "w", encoding='utf-8')
    with open("../../data/smp/training.txt", "r", encoding='utf-8') as f:
        lines = f.readlines()
        cnt = len(lines)
        for i, line in enumerate(lines):
            print(i)
            data = json.loads(line)
            label = get_label(data['标签'])
            seg_text = jieba.cut(data['内容'].replace("\t", " ").replace("\n", " "))
            outline = " ".join(seg_text)
            outline = outline + "\t" + label + "\n"
            if i <= cnt * 0.8:
                ftrain.write(outline)
                ftrain.flush()
            else:
                ftest.write(outline)
                ftest.flush()
    ftrain.close()
    ftest.close()

load_data()
