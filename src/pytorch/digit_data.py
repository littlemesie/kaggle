# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

def load_data_train(train_tatio= 0.8):
    """数据训练加载"""
    pd_train = pd.read_csv('../../data/digit-recognizer/train.csv', header=0)
    trainY = pd_train.label.values.tolist()
    print(len(trainY))
    trainX = pd_train.drop('label', axis=1).values.reshape(pd_train.shape[0], 1, 28, 28)
    # trainX = pd_train.drop('label', axis=1).values.reshape(-1, 28, 28, 1)
    # 切分为训练集和测试集
    row = pd_train.shape[0]
    split_num = int(train_tatio * row)
    data_trainY = trainY[:split_num]
    pd_data_testY = trainY[split_num:]
    data_trainX = trainX[:split_num]
    pd_data_testX = trainX[split_num:]
    # show(trainX[10])
    return data_trainX, data_trainY, pd_data_testX, pd_data_testY

def load_data_pre():
    """测试数据集"""
    pd_pre = pd.read_csv('../../data/digit-recognizer/test.csv', header=0)
    testX = pd_pre.values.reshape(pd_pre.shape[0], 1, 28, 28)
    show(testX[0])

def show(x):
    x = x / 255
    print(x[0, :, :])
    plt.imshow(x[0, :, :])
    plt.show()

if __name__ == '__main__':
    load_data_pre()