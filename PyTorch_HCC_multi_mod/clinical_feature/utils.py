import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pypinyin


def correlation_analysis(data, correlation_threshold):
    corr_matrix = data.corr()
    print('corr_matrix', corr_matrix)
    # sns.heatmap(corr_matrix)
    # plt.show()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # for column in upper.columns:
    #     print(column)
    #     print(upper[column])
    # print('upper', upper)
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
    return to_drop


def preprocessing_num(train_df, test_df, features_in):
    for i in range(len(features_in)):
        feature = np.array(train_df[features_in[i]].tolist())
        mean_v = np.max(feature)
        std_v = np.std(feature)
        feature_tr = (feature-mean_v)/std_v
        train_df[features_in[i]] = list(feature_tr)

        feature_ts = np.array(test_df[features_in[i]].tolist())
        feature_ts = list((feature_ts - mean_v) / std_v)
        test_df[features_in[i]] = feature_ts
    return train_df, test_df


def pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s