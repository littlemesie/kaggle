# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/1/6 19:24
@summary:
"""
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import zipfile

from sklearn import model_selection

SENTIMENT_LABELS = [
    "negative", "somewhat negative", "neutral", "somewhat positive", "positive"
]

# 添加一列
def add_readable_labels_column(df, sentiment_value_column):
    df["SentimentLabel"] = df[sentiment_value_column].replace(range(5), SENTIMENT_LABELS)

# 获取数据
def get_data(validation_set_ratio=0.01):
    train_df = pd.read_csv("../data/movie/train.tsv", sep='\t')
    test_df = pd.read_csv("../data/movie/test.tsv", sep='\t')
    # Add a human readable label.
    add_readable_labels_column(train_df, "Sentiment")

    # We split by sentence ids, because we don't want to have phrases belonging
    # to the same sentence in both training and validation set.
    train_indices, validation_indices = model_selection.train_test_split(
        np.unique(train_df["SentenceId"]),
        test_size=validation_set_ratio,
        random_state=0)

    validation_df = train_df[train_df["SentenceId"].isin(validation_indices)]
    train_df = train_df[train_df["SentenceId"].isin(train_indices)]
    print("Split the training data into %d training and %d validation examples." %
            (len(train_df), len(validation_df)))

    return train_df, validation_df, test_df

def model(train_df, validation_df, test_df):
    # Training input on the whole training set with no limit on training epochs.
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["Sentiment"], num_epochs=None, shuffle=True)

    # Prediction on the whole training set.
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["Sentiment"], shuffle=False)
    # Prediction on the validation set.
    predict_validation_input_fn = tf.estimator.inputs.pandas_input_fn(
        validation_df, validation_df["Sentiment"], shuffle=False)
    # Prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, shuffle=False)

    embedded_text_feature_column = hub.text_embedding_column(
        key="Phrase",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",
        trainable=True)

    # We don't need to keep many checkpoints.
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=1)

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[250, 50],
        feature_columns=[embedded_text_feature_column],
        n_classes=5,
        config=run_config,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    validation_eval_result = estimator.evaluate(input_fn=predict_validation_input_fn)

    print("Training set accuracy: {accuracy}".format(**train_eval_result))
    print("Validation set accuracy: {accuracy}".format(**validation_eval_result))

if __name__ == '__main__':
    train_df, validation_df, test_df = get_data()
    print(train_df.head())