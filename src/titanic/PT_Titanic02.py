# -*- coding:utf-8 -*-
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def data_process():
    """数据预处理"""
    train = pd.read_csv("../data/titanic/train.csv")
    test = pd.read_csv("../data/titanic/test.csv")
    # 删除Name、Ticket、Cabin
    train = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    combine = [train, test]
    # 数值化
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
        dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
        dataset['P1'] = np.array(dataset['Pclass'] == 1).astype(np.int32)
        dataset['P2'] = np.array(dataset['Pclass'] == 2).astype(np.int32)
        dataset['P3'] = np.array(dataset['Pclass'] == 3).astype(np.int32)
        dataset['E1'] = np.array(dataset['Embarked'] == "S").astype(np.int32)
        dataset['E2'] = np.array(dataset['Embarked'] == "C").astype(np.int32)
        dataset['E3'] = np.array(dataset['Embarked'] == "Q").astype(np.int32)

    for dataset in combine:
        del dataset['Pclass']
        del dataset['Embarked']

    inp_train = np.array(combine[0][['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'P1', 'P2', 'P3', 'E1', 'E2', 'E3']])
    inp_test = np.array(combine[1][['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'P1', 'P2', 'P3', 'E1', 'E2', 'E3']])
    out_target = np.array(combine[0][['Survived']])
    test_passenger_id = combine[1][['PassengerId']]
    return inp_train, inp_test, out_target, test_passenger_id



# 模型构建
learning_rate = 0.05
weight_decay = 0.0001
train_epochs = 25
test_epochs = 1

class TitanicNet(nn.Module):

    def __init__(self):
        super(TitanicNet, self).__init__()
        self.l1 = nn.Linear(11, 32)
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
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train(inp_train, out_target):

    for epoch in range(train_epochs):

        correct = 0
        total = 0
        total_train_loss = 0
        op = []
        inputs = torch.Tensor(inp_train)
        labels = torch.Tensor(out_target)
        for i in range(0, len(labels)):
            if (labels[i] == 0):
                op.append([1, 0])
            else:
                op.append([0, 1])


        optimizer.zero_grad()
        outputs = model(inputs)
        target = torch.Tensor(op)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss = total_train_loss + loss.item()
        total += target.size(0)

        _, index = torch.max(outputs, 1)
        index = [float(r) for r in index]
        labels = [float(r) for r in labels]
        for r in range(len(index)):
            if (index[r] == labels[r]):
                correct += 1

        model_accuracy = (100 * float(correct)) / float(total)
        train_loss = float(total_train_loss)

        print('----------------**----------------')
        print("Epoch : ", epoch)
        print('Accuracy of the model : ', model_accuracy)
        print('Correct :', correct)
        print('total :', total)
        print('train_loss :', train_loss)

def test(inp_test, test_passenger_id):

    for epoch in range(test_epochs):

        t = []
        q = torch.Tensor(inp_test)

        optimizer.zero_grad()
        y_pred = model(q)

        for m in range(0, len(q)):
            if (y_pred.data[m][0] > y_pred.data[m][1]):
                t.append('0')
            else:
                t.append('1')

        df_pred = pd.DataFrame({'output': t})

        pred = pd.DataFrame({'PassengerId': test_passenger_id.PassengerId, 'Survived': df_pred.output })
        pred.to_csv("Submission.csv", index=False)

if __name__ == '__main__':
    inp_train, inp_test, out_target, test_passenger_id = data_process()

    train(inp_train, out_target)

    # test(inp_test, test_passenger_id)