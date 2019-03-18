# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/3/16 13:14
@summary:
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch import titanic_data

# 模型构建
learning_rate = 0.05
weight_decay = 0.0001
train_epochs = 25
test_epochs = 1

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        output = F.relu(self.hidden(x))
        y = self.out(output)
        return y


net = Net(n_feature=9, n_hidden=20, n_output=2)
criterion = torch.nn.CrossEntropyLoss()

# 更新网络的权重
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

train, target, test = titanic_data.result_data()

inp_train = np.asarray(train)
out_target = np.asarray(target)
inp_test = np.asarray(test)


def train(train, target, test):

    timer = time.time()
    for epoch in range(train_epochs):
        correct = 0
        total = 0
        total_train_loss = 0
        # op = []
        q = torch.Tensor(train).float()
        t = torch.Tensor(target).long()
        # 清零梯度缓存
        optimizer.zero_grad()
        q = Variable(q)
        # 喂给 net 训练数据
        y_pred = net(q)
        y = Variable(t)
        # 计算两者的误差
        loss = criterion(y_pred, y)
        # 反向传播权重
        loss.backward()
        # 更新参数
        optimizer.step()

        total_train_loss = total_train_loss + loss.data
        total += y.size(0)
        # print(total_train_loss)

        _, index = torch.max(y_pred, 1)
        index = [float(r) for r in index]
        t = [float(r) for r in t]
        for r in range(len(index)):
            if (index[r] == t[r]):
                correct += 1

        model_accuracy = (100 * float(correct)) / float(total)
        train_loss = float(total_train_loss)

        print('Time for epoch : ', time.time() - timer)
        print("Epoch : ", epoch)
        print('Accuracy of the model : ', model_accuracy)
        print('Correct :', correct)
        print('total :', total)

if __name__ == '__main__':
    train(inp_train, out_target, inp_test)