# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
"""
titanic 数据处理
"""

def read_csv():
    """读取文件"""
    pd_train = pd.read_csv("../../data/titanic/train.csv")
    pd_test = pd.read_csv("../../data/titanic/test.csv")
    # 删除Ticket和Cabin
    train = pd_train.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
    test = pd_test.drop(['Ticket', 'Cabin'], axis=1)
    train['SibSp'] = train['SibSp'].fillna(0)
    test['SibSp'] = test['SibSp'].fillna(0)
    train['Parch'] = train['Parch'].fillna(0)
    test['Parch'] = test['Parch'].fillna(0)
    train['Pclass'] = train['Pclass'].fillna(0)
    test['Pclass'] = test['Pclass'].fillna(0)

    train = transform_features(train)
    test = transform_features(test)
    return train, test

def transform_features(data):
    """组合特征"""
    df = handle_title(data)
    df = handle_age(df)
    df = handle_sex(df)
    df = handle_embarked(df)
    df = handle_fare(df)

    return df

def handle_title(data):
    """处理name"""
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(
                ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)
    data = data.drop(['Name'], axis=1)
    return data


def handle_age(data):
    """age"""
    data['Age'] = data['Age'].fillna(data['Age'].dropna().mean())
    data.loc[data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    return data

def handle_sex(data):
    """sex"""
    data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)
    data['Sex'] = data['Sex'].fillna(0)
    return data

def handle_embarked(data):
    """Embarked"""
    freq_port = data.Embarked.dropna().mode()[0]
    data['Embarked'] = data['Embarked'].fillna(freq_port)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    return data

def handle_fare(data):
    """Fare"""
    data['Fare'].fillna(data['Fare'].dropna().median(), inplace=True)
    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)

    return data


def train_data(train,train_tatio= 0.8):
    """处理训练数据"""
    # 切分为训练集和测试集
    trainY = train['Survived']
    trainX = train.drop('Survived', axis=1)
    row = train.shape[0]
    split_num = int(train_tatio * row)
    data_trainX = trainX[:split_num]
    data_testX = trainX[split_num:]
    data_trainY = trainY[:split_num]
    data_testY = trainY[split_num:]

    return data_trainX, data_testX, data_trainY, data_testY


if __name__ == '__main__':
    train, test = read_csv()

