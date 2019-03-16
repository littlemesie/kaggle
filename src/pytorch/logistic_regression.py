# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/3/16 13:14
@summary:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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


net = Net(n_feature=2, n_hidden=10, n_output=2)
criterion = nn.MSELoss()

# 更新网络的权重
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate,
                             weight_decay=weight_decay)