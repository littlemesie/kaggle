# -*- coding:utf-8 -*-
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# 数据预处理
df_train = pd.read_csv("../data/titanic/train.csv")
df_test = pd.read_csv("../data/titanic/test.csv")

# 用平均年龄填充空值
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())

df_train['Has_Cabin'] = df_train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
df_test['Has_Cabin'] = df_test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']


def simplify_ages(df):
	df['Age'] = df['Age'].astype(int)
	bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
	group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
	categories = pd.cut(df['Age'], bins, labels=group_names)
	df['Age'] = categories.cat.codes
	return df


def simplify_fares(df):
	df['Fare'] = df.Fare.fillna(-0.5)
	bins = (-1, 0, 8, 15, 31, 1000)
	group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
	categories = pd.cut(df['Fare'], bins, labels=group_names)
	df['Fare'] = categories.cat.codes
	return df

def simplify_sex(df):
	df['Sex'] = pd.Categorical(df['Sex'])
	df['Sex'] = df['Sex'].cat.codes
	return df

def simplify_embarked(df):
    df['Embarked'] = pd.Categorical(df['Embarked'])
    df['Embarked'] = df['Embarked'].cat.codes + 1
    return df

def transform_features(df):
	df = simplify_ages(df)
	df = simplify_fares(df)
	df = simplify_sex(df)
	df = simplify_embarked(df)
	return df

df_train = transform_features(df_train)
df_test = transform_features(df_test)

inp_train = df_train.drop(['PassengerId','Name','Ticket','Survived','Cabin'], axis=1)
out_train = df_train['Survived']
inp_test = df_test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

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

    def num_flat_features(self, x):
        output = self.forward_once(x)
        return output



model = TitanicNet()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                             weight_decay=weight_decay)

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
        q = torch.FloatTensor(rest_inp_train)
        t = torch.FloatTensor(rest_out_train)
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
        loss.backward()
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

        pred = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': df_pred.output })
        pred.to_csv("Submission.csv", index=False)

train(inp_val,rest_inp_train,out_val,rest_out_train)
test(inp_test)