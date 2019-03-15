# -*- coding:utf-8 -*-
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch import titanic_data


"""
线性回归
"""
# 模型构建
learning_rate = 0.05
weight_decay = 0.0001
train_epochs = 25
test_epochs = 1

class TitanicNet(nn.Module):

    def __init__(self):
        super(TitanicNet, self).__init__()
        self.l1 = nn.Linear(9, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 8)
        self.l4 = nn.Linear(8, 2)

    def forward(self, x):
        output = F.relu(self.l1(x))
        output = F.relu(self.l2(output))
        output = F.relu(self.l3(output))
        output = F.sigmoid(self.l4(output))
        return output



model = TitanicNet()
# 损失函数 均方误差
criterion = nn.MSELoss()
# 创建优化器（optimizer）
# Adam 一种基于一阶梯度的随机目标函数优化算法一种基于一阶梯度的随机目标函数优化算法
# 更新网络的权重
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                             weight_decay=weight_decay)

inp_train, out_train, inp_test = titanic_data.result_data()

out_train = np.asarray(out_train)
inp_train = np.asarray(inp_train)
inp_test = np.asarray(inp_test)

val_size = 100

inp_val = inp_train[:val_size]
rest_inp_train = inp_train[val_size:]

out_val = out_train[:val_size]
rest_out_train = out_train[val_size:]


def train(inp_val, rest_inp_train, out_val, rest_out_train):
    timer = time.time()

    for epoch in range(train_epochs):
        i = 0
        correct = 0
        total = 0
        total_train_loss = 0
        op = []
        q = torch.Tensor(rest_inp_train).float()
        t = torch.Tensor(rest_out_train).float()
        for i in range(0, len(t)):
            if (t[i] == 0):
                op.append([1, 0])
            else:
                op.append([0, 1])
        # 清零梯度缓存
        optimizer.zero_grad()
        q = Variable(q)
        # 喂给 net 训练数据
        y_pred = model(q)
        target = Variable(torch.Tensor(op))
        # 计算两者的误差
        loss = criterion(y_pred, target)
        # 反向传播权重
        loss.backward()
        # 更新参数
        optimizer.step()

        # print(target,y_pred)
        total_train_loss = total_train_loss + loss.data
        total += target.size(0)
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

        # writer.add_scalar('Train_loss', train_loss, epoch + 1)
        # writer.add_scalar('Train_Accuracy', model_accuracy, epoch + 1)

        timer = time.time()

        i = 0
        correct = 0
        total = 0
        total_valid_loss = 0
        op = []
        q = torch.FloatTensor(inp_val)
        t = torch.FloatTensor(out_val)
        for i in range(0, len(t)):
            if (t[i] == 0):
                op.append([1, 0])
            else:
                op.append([0, 1])
        optimizer.zero_grad()
        q = Variable(q)
        y_pred = model(q)
        target = Variable(torch.Tensor(op))
        loss = criterion(y_pred, target)

        total_valid_loss = total_valid_loss + loss.data
        total += target.size(0)
        # print(total_train_loss)

        _, index = torch.max(y_pred, 1)
        index = [float(r) for r in index]
        t = [float(r) for r in t]
        for r in range(len(index)):
            if (index[r] == t[r]):
                correct += 1

        model_accuracy = (100 * float(correct)) / float(total)
        valid_loss = float(total_valid_loss)

        print('Time for epoch : ', time.time() - timer)
        print("Epoch : ", epoch)
        print('Accuracy of the model : ', model_accuracy)
        print('Correct :', correct)
        print('total :', total)

        # writer.add_scalar('Valid_loss', valid_loss, epoch + 1)
        # writer.add_scalar('Valid_Accuracy', model_accuracy, epoch + 1)

        timer = time.time()


def test(inp_test):
    timer = time.time()
    train, test = titanic_data.read_csv()
    for epoch in range(test_epochs):

        t = []
        q = torch.FloatTensor(inp_test)

        optimizer.zero_grad()
        q = Variable(q)
        y_pred = model(q)

        for m in range(0, len(q)):
            if (y_pred.data[m][0] > y_pred.data[m][1]):
                t.append('0')
            else:
                t.append('1')

        df_pred = pd.DataFrame({'output': t})

        pred = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': df_pred.output })
        pred.to_csv("Submission.csv", index=False)

if __name__ == '__main__':
    train(inp_val, rest_inp_train, out_val, rest_out_train)