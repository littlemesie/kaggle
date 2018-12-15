# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/13 23:40
@summary:
"""

# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


# 数据预处理
train = pd.read_csv("../data/titanic/train.csv")
test = pd.read_csv("../data/titanic/test.csv")
# 删除Name、Ticket、Cabin
train = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)

test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
combine = [train, test]
# 性别数值化
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

data_train = np.array(combine[0][['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'P1', 'P2', 'P3', 'E1', 'E2', 'E3']])
data_test = combine[1][['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'P1', 'P2', 'P3', 'E1', 'E2', 'E3']]

data_target = np.array(combine[0][['Survived']])

# 模型构建

x = tf.placeholder('float', shape=[None, 11])
y = tf.placeholder('float', shape=[None, 1])
weiget = tf.Variable(tf.random_normal([11, 1]))
bias = tf.Variable(tf.random_normal([1]))
output = tf.matmul(x, weiget) + bias
pred = tf.cast(tf.sigmoid(output) > 0.5, tf.float32)
# 模型构建
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output))
# 使用梯度下降
train_step = tf.train.GradientDescentOptimizer(0.0003).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

# 运行
sess = tf.Session()
sess.run(tf.global_variables_initializer())
loss_train = []
train_acc = []
test_acc = []

for i in range(25000):
    index = np.random.permutation(len(data_target))
    data_train = data_train[index]
    data_target = data_target[index]
    for n in range(len(data_target) // 100 + 1):
        batch_xs = data_train[n * 100: n * 100 + 100]
        batch_ys = data_target[n * 100: n * 100 + 100]
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

    if i % 1000 == 0:
        loss_tmp = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
        loss_train.append(loss_tmp)
        train_acc_tmp = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        train_acc.append(train_acc_tmp)
        print(loss_tmp, train_acc_tmp)
