# Wendy Li wendy.li@philips.com
# Data Science, ATS, MSC

import numpy as np
import pandas as pd
import sklearn.metrics
# import pingouin as pg
from scipy import stats
# import statsmodels.formula.api as smf
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter
import re


# test_type: two-sided, less, greater
def two_independent_sample_test(v_1, v_2, thresh_p, test_type='two-sided'):
    # print("#########################")
    mean_1 = np.mean(v_1)
    mean_2 = np.mean(v_2)
    std_1 = np.std(v_1)
    std_2 = np.std(v_2)
    print('v_1', mean_1, std_1)
    print('v_2', mean_2, std_2)
    # per_1 = (np.percentile(v_1, 2.5), np.percentile(v_1, 97.5))
    # per_2 = (np.percentile(v_2, 2.5), np.percentile(v_2, 97.5))
    # normal distribution test
    s_t_1 = stats.shapiro(v_1)[1] < 0.05
    s_t_2 = stats.shapiro(v_2)[1] < 0.05
    print('Pass normal test:', s_t_1 == s_t_2 == False)
    # test whether all input samples are from populations with equal variances
    stat, p = stats.levene(v_1, v_2)
    print('p value of homogeneity of variance:', p > 0.05)
    test_type_select = 't test'
    if (s_t_1 == s_t_2 == False) & (p > 0.05):
        print('feature can be analyzed by t test')
        stat, p_v = stats.ttest_ind(v_1, v_2, alternative=test_type)
        print('p_value of t_test', p_v)
        print('stat', stat)
    else:
        test_type_select = 'Mann-Whitney U test'
        # rank sum test
        print('feature can be analyzed be Mann-Whitney test')
        # w, p_v = stats.wilcoxon(v_1, v_2)
        # Compute the Mann-Whitney rank test on samples x and y.
        w, p_v = stats.mannwhitneyu(v_1, v_2, use_continuity=True, alternative=test_type)
        print(w, p_v)
        # w, p_v = stats.ranksums(v_1, v_2)
        # print(w, p_v)
        print('p_value of Mann-Whitney test', round(p_v, 3))
        # p_v = stats.kruskal(v_0, v_1, v_2)[1]
    if p_v < thresh_p:
        print('#'*200)
        print('Feature that has difference among two groups')
        print('p_value is smaller than thresh_p', p_v < 0.05)
    else:
        print('p_value is smaller than thresh_p', p_v < 0.05)
    print(type(mean_1))
    return p_v, test_type_select


# histogram of different groups
def plot_hist(feature_1, feature_2, title, root_path):
    plt.hist(feature_1, color='g', alpha=0.3, label="label 0", density=1)
    plt.hist(feature_2, color='r', alpha=0.3, label="label 1", density=1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(root_path, title+'_hist.png'))
    plt.close()


