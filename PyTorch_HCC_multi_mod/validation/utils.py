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
    feature = train_df[features_in]
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(feature)

    feature = min_max_scaler.transform(feature)
    # feature = pd.DataFrame(feature)
    # feature.columns = features_in
    train_df[features_in] = feature

    feature_test = test_df[features_in]
    feature_test = min_max_scaler.transform(feature_test)
    # feature_test = pd.DataFrame(feature_test)
    # feature_test.columns = features_in
    test_df[features_in] = feature_test
    return train_df, test_df


def pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s