# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/2 14:52
"""
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt

# 获取数据
def get_data():
    train = pd.read_csv("../data/titanic/train.csv")
    test = pd.read_csv("../data/titanic/test.csv")
    # 删除Ticket，Cabin两个特征
    train = train.drop(['Ticket', 'Cabin'], axis=1)
    test = test.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train, test]
    return train,test,combine

# 数据处理
def data_handle(train,test,combine):
    """"""

def sex(train, test, combine):
    """性别处理"""
    # 获取名字里有Miss Mr 组成新特征
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    # 姓名处理
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                                                     'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        dataset['Title'] = dataset['Title'].map(title_mapping)
        # 将空值置为0
        dataset['Title'] = dataset['Title'].fillna(0)
        # 把性别向量化
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
    # 交叉表是用于统计分组频率
    # pd.crosstab(train['Title'], train['Sex'])
    # 删除Name及PassengerId
    train = train.drop(['Name', 'PassengerId'], axis=1)
    test = test.drop(['Name'], axis=1)
    combine = [train, test]
    return train, test, combine

def age(train, test, combine):
    """年龄处理"""
    for dataset in combine:
        dataset['Age'] = dataset['Age'].fillna(0)
        # 把年龄变成数字
        dataset['Age'] = dataset['Age'].astype(int)

    train_ave_age = int(combine[0]['Age'].mean())
    combine[0]['Age'] = combine[0]['Age'].fillna(train_ave_age)
    test_ave_age = int(combine[1]['Age'].mean())
    combine[1]['Age'] = combine[1]['Age'].fillna(test_ave_age)
    # 把年龄切分五分
    train['AgeBand'] = pd.cut(train['Age'], 5)
    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age']
    train = train.drop(['AgeBand'], axis=1)

    combine = [train, test]
    return train, test, combine

def family(train, test, combine):
    """家庭特征"""
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    # 删除特征
    train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    print(train.head())
    combine = [train, test]

    return train, test, combine

def embarked(train, test, combine):
    freq_port = train.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    combine = [train, test]
    return train, test, combine

def fare(train, test, combine):
    # 处理test里缺失的数据，用中位数代替
    test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
    # 处理Fare
    train['FareBand'] = pd.qcut(train['Fare'], 4)

    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train = train.drop(['FareBand'], axis=1)
    combine = [train, test]

if __name__ == '__main__':

    train,test,combine = get_data()
    train, test, combine =sex(train, test, combine)
    train, test, combine = age(train, test, combine)
    train, test, combine = family(train, test, combine)