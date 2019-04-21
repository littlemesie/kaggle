# _*_coding:utf-8 _*_
import fasttext
"""
测试
"""
# texts = ['分析 中国 经济 ， 要 看 这 艘 大船 方向 是否 正确 ， 动力 是否 强劲 ， 潜力 是否 充沛 。 只要 投资者 全面 了解 中国 改革开放 以来 的 经济 发展 历程 、 近期 中国 为 促进 经济 持续 稳定增长 制定 的 战略 以及 中国 经济 各项 数据 和 趋势 ， 就 会 作出 正确 判断 。',
#           ]
# print(texts)
#load训练好的模型
classifier = fasttext.load_model('model/smp.model.bin', label_prefix='__label__')

with open("../../../data/smp/test.txt", "r", encoding='utf-8') as f:
    lines = f.readlines()
    sum = len(lines)
    acc_count = 0
    for line in lines:
        texts = []
        line = line.split("__label__")
        text = [line[0].strip()]
        try:
            label = line[1].strip()
        except:
            continue

        try:
            label_ = classifier.predict(text)
        except:
            continue
        # print(label)
        # print(label_[0][0])
        if label == label_[0][0]:
            acc_count += 1
        # break
    print(acc_count)
    accuracy = acc_count / sum
    print('accuracy:%.4f' % accuracy)

# labels = classifier.predict(texts)
# print(labels)