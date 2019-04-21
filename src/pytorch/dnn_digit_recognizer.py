# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(1)


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,)) # 拉平
    x = torch.from_numpy(x)
    return x

# 使用内置函数下载 mnist 数据集
train_set = dsets.mnist.MNIST('../../data/mnist/', train=True, transform=data_tf, download=True)
test_set = dsets.mnist.MNIST('../../data/mnist/', train=False, transform=data_tf, download=True)


# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.1               # learning rate


train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_data = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)




net = nn.Sequential(
                nn.Linear(784, 400),
                nn.ReLU(),
                nn.Linear(400, 200),
                nn.ReLU(),
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Linear(100, 10)
            )
print(net)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_func = nn.CrossEntropyLoss()

# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(20):
    train_loss = 0
    train_acc = 0
    # train_num_correct = 0
    net.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = net(im)
        loss = loss_func(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.data
        # 计算分类的准确率
        _, pred = out.max(1)
        # print(pred == label)
        num_correct = torch.sum((pred == label))

        acc = np.array(num_correct) / im.shape[0]
        train_acc += acc

    train_acc = train_acc / len(train_data)

    losses.append(train_loss / len(train_data))
    acces.append(train_acc)
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    test_num_correct = 0
    net.eval()  # 将模型改为预测模式
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = loss_func(out, label)
        # 记录误差
        eval_loss += loss.data
        # 记录准确率
        _, pred = out.max(1)
        num_correct = torch.sum((pred == label))
        acc = np.array(num_correct) / im.shape[0]
        eval_acc += acc

    eval_acc = eval_acc / len(test_data)

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc)
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc,
                  eval_loss / len(test_data), eval_acc))

plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)