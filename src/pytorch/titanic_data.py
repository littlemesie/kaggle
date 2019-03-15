# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
"""
titanic 数据处理
"""

def read_csv():
    """读取文件"""
    train = pd.read_csv("../../data/titanic/train.csv")
    test = pd.read_csv("../../data/titanic/test.csv")
    return train, test

def simple_process(train, test):
    """大概、简单处理"""
    # 用平均年龄填充空值
    train['Age'] = train['Age'].fillna(train['Age'].mean())
    test['Age'] = test['Age'].fillna(test['Age'].mean())
    # 处理Cabin
    train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    # 组合SibSp和Parch
    train['FamilySize'] = train['SibSp'] + train['Parch']
    test['FamilySize'] = test['SibSp'] + test['Parch']
    return train, test

def simplify_ages(df):
    """年龄处理"""
    df['Age'] = df['Age'].astype(int)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df['Age'], bins, labels=group_names)
    df['Age'] = categories.cat.codes
    return df


def simplify_fares(df):
    """"""
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
    """组合特征"""
    df = simplify_ages(df)
    df = simplify_fares(df)
    df = simplify_sex(df)
    df = simplify_embarked(df)
    return df

def result_data():
    train, test = read_csv()
    train, test = simple_process(train, test)

    df_train = transform_features(train)
    df_test = transform_features(test)

    inp_train = df_train.drop(['PassengerId','Name','Ticket','Survived','Cabin'], axis=1)
    out_train = df_train['Survived']
    inp_test = df_test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

    return inp_train, out_train, inp_test

if __name__ == '__main__':
    train, test = read_csv()
    print(train['Cabin'])
