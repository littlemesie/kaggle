# -*- coding:utf-8 -*-
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class CustomedDataSet(Dataset):

    def __init__(self, pd_data, data_type=True):
        self.data_type = data_type
        if self.data_type:
            trainX = pd_data
            trainY = trainX.label.as_matrix().tolist()
            trainX = trainX.drop('label', axis=1).as_matrix().reshape(trainX.shape[0], 1, 28, 28)
            self.datalist = trainX
            self.labellist = trainY
        else:
            testX = pd_data
            testX = testX.as_matrix().reshape(testX.shape[0], 1, 28, 28)
            self.datalist = testX

    def __getitem__(self, index):
        if self.data_type:
            return torch.Tensor(self.datalist[index].astype(float)), self.labellist[index]
        else:
            return torch.Tensor(self.datalist[index].astype(float))

    def __len__(self):
        return self.datalist.shape[0]

def load_data_train(train_tatio= 0.8):
    """数据训练加载"""
    pd_train = pd.read_csv('../../data/digit-recognizer/train.csv', header=0)
    trainY = pd_train.label.values.tolist()
    trainX = pd_train.drop('label', axis=1).values.reshape(pd_train.shape[0], 1, 28, 28)
    # trainX = pd_train.drop('label', axis=1).values.reshape(-1, 28, 28, 1)
    # 切分为训练集和测试集
    row = pd_train.shape[0]
    split_num = int(train_tatio * row)
    data_trainX = trainX[:split_num]
    data_testX = trainX[split_num:]
    data_trainY = trainY[:split_num]
    data_testY = trainY[split_num:]
    # show(trainX[10])
    return data_trainX, data_trainY, data_testX, data_testY

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