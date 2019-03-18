# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/3/18 22:39
@summary:
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# 加载数据
def load_data():
    # 使用 pandas 打开
    data = pd.read_csv('../../data/digit-recognizer/train.csv')
    data1 = pd.read_csv('../../data/digit-recognizer/test.csv')

    train_data = data.values[0:, 1:]  # 读入全部训练数据
    train_label = data.values[0:, 0]
    test_data = data1.values[0:, 0:]  # 测试全部测试个数据
    return train_data, train_label, test_data

# 模型训练
def knn(trainData, trainLabel, testData):
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(trainData, trainLabel.ravel())
    testLabel = knn_clf.predict(testData)
    print(testLabel)

if __name__ == '__main__':
    train_data, train_label, test_data = load_data()
    knn(train_data, train_label, test_data)