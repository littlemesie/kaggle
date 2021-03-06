#!/usr/bin/python
# coding: utf-8

import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(1)


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


class CNN(nn.Module):
    """创建CNN模型"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            # output shape (16, 28, 28)  28=(width+2*padding-kernel_size)/stride+1
            nn.Conv2d(
                in_channels=1,    # 输入信号的通道数
                out_channels=16,  # 卷积后输出结果的通道数
                kernel_size=5,    # 卷积核的形状
                stride=1,         # 卷积核得步长
                padding=2,        # 处理边界时在每个维度首尾补0数量 padding=(kernel_size-1)/2 if stride=1
            ),
            nn.ReLU(),  # activation
            # output shape (16, 14, 14)  14=(width+0*padding-dilation*(kernel_size-1)-1)/stride+1
            # dilation=1 默认值为0
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化操作时的窗口大小
        )
        # output shape (32, 7, 7)
        self.conv2 = nn.Sequential(      # input shape (1, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2, 2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 扁平化操作
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x  # return x for visualization


def load_data_train(train_tatio= 0.8):
    """数据加载"""
    pd_train = pd.read_csv('../../data/digit-recognizer/train.csv', header=0)
    # 切分为训练集和测试集
    row = pd_train.shape[0]
    split_num = int(train_tatio * row)
    pd_data_train = pd_train[:split_num]
    pd_data_test = pd_train[split_num:]
    # 转换成 torch 能识别的 Dataset
    data_train = CustomedDataSet(pd_data_train, data_type=True)
    data_test = CustomedDataSet(pd_data_test, data_type=True)



    # 数据读取器 (Data Loader)
    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    loader_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_test = DataLoader(dataset=data_test, batch_size=pd_data_test.shape[0], shuffle=True)
    return loader_train, loader_test


def load_data_pre():
    pd_pre = pd.read_csv('../../data/digit-recognizer/test.csv', header=0)
    data_pre = CustomedDataSet(pd_pre, data_type=False)
    loader_pre = DataLoader(dataset=data_pre, batch_size=BATCH_SIZE, shuffle=True)
    return loader_pre


def optimizer_lossfunction(cnn,lr=0.0001):
    """设置优化器和损失函数"""
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr, betas=(0.9, 0.99))
    loss_func = nn.CrossEntropyLoss()
    return optimizer, loss_func

def train_model(cnn, optimizer, loss_func, loader_train, loader_test):
    """训练模型"""
    # plt.ion()
    EPOCH = 10
    # training and testing
    for epoch in range(EPOCH):
        num = 0
        for step, (x, y) in enumerate(loader_train):
            b_x = Variable(x)
            b_y = Variable(y)
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                num += 1
                for _, (x_t, y_test) in enumerate(loader_test):
                    test_sum = y_test.size(0)
                    accuracy_sum = 0
                    x_test = Variable(x_t)
                    test_output, last_layer = cnn(x_test)
                    pred_y = torch.max(test_output, 1)[1].data.squeeze()

                    for i in range(test_sum):
                        if pred_y[i] == y_test[i]:
                            # print('预测:',pred_y[i], '正确:', y_test[i])
                            accuracy_sum += 1
                    accuracy = accuracy_sum / test_sum
                    print('Epoch:', epoch, '| Num: ',  num, '| Step: ', step, '| train loss: %.4f' % loss.data, '| test accuracy: %.4f' % accuracy)
            # if step % 50 == 0:
            #     num += 1
            #     for _, (x_t, y_test) in enumerate(loader_test):
            #         x_test = Variable(x_t)
            #         test_output, last_layer = cnn(x_test)
            #         pred_y = torch.max(test_output, 1)[1].data.squeeze()
            #         accuracy = sum(pred_y == y_test) / float(y_test.size(0))
            #         print('Epoch:', epoch, '| Num: ',  num, '| Step: ',  step, '| train loss: %.4f' % loss.data, '| test accuracy: %.4f' % accuracy)
                    # 可视化展现
                    # show(last_layer, y_test)
    # plt.ioff()
    return cnn


def model_cnn():
    """模型训练"""
    # 1.加载数据
    loader_train, loader_test = load_data_train(0.8)
    # 2.创建CNN模型
    cnn = CNN()
    # 3. 设置优化器和损失函数
    optimizer, loss_func = optimizer_lossfunction(cnn, 0.0001)
    # 4. 训练模型
    cnn = train_model(cnn, optimizer, loss_func, loader_train, loader_test)
    return cnn


def prediction(cnn, loader_pre):
    """预测"""
    for step, (x, y) in enumerate(loader_pre):
        b_x = Variable(x)  # batch x
        test_output, _ = cnn(b_x)

        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        print(pred_y, 'prediction number')

    return pred_y


if __name__ == "__main__":
    global BATCH_SIZE, TRAIN_TATIO, LR, momentum
    BATCH_SIZE = 50

    # 训练模型
    file_path = '../../data/digit-recognizer/net.pkl'
    cnn = model_cnn()
    torch.save(cnn, file_path)

    # 预测数据
    loader_pre = load_data_pre()
    cnn = torch.load(file_path)
    pre_data = prediction(cnn, loader_pre)
    print(pre_data)
