# Wendy Li wendy.li@philips.com
# Data Science, ATS, MSC


from scipy import stats
from statsmodels.stats.anova import anova_lm
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ranksums
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# Multiplt comparison
def multicomp(score, group, feature_name):
    mc_results = pairwise_tukeyhsd(score, group)
    print('type', type(mc_results.summary()))
    print(mc_results.summary().as_csv())
    return mc_results.summary().as_csv()


def p_value_3_feature(v_0, v_1, v_2):
    v_col = v_0 + v_1 + v_2
    labels = [0 for i in v_0] + [1 for i in v_1] + [2 for i in v_2]
    # normal test
    s_t_0 = stats.shapiro(v_0)[1] < 0.05
    s_t_1 = stats.shapiro(v_1)[1] < 0.05
    s_t_2 = stats.shapiro(v_2)[1] < 0.05
    print()
    print('Pass normal test:', s_t_0 == s_t_1 == s_t_2 == False)

    # test whether all input samples are from populations with equal variances
    stat, p = stats.levene(v_0, v_1, v_2)
    print('p value of homogeneity of variance:', p > 0.05)
    df = pd.DataFrame({'x': v_col, 'y': labels})
    if (s_t_0 == s_t_1 == s_t_2 == False) & (p > 0.05):
        normal_dist = True
        print('feature can be analized by variance analysis')
        # print('s_t_0', s_t_0)
        # print('s_t_1', s_t_1)
        # print('s_t_2', s_t_2)

        # print('v_0', len(v_0), v_0)
        # print('v_1', len(v_1), v_1)
        # print('v_2', len(v_2), v_2)

        # ANOVA
        model = smf.ols('y~x', data=df).fit()
        anovat = anova_lm(model, typ=2)
        print('anovat', anovat)
        # print(type(anovat), anovat.columns)
        p_v = anovat['PR(>F)'].tolist()[0]
        print('p_value of ANOVA', p_v)

        # print('anovat', anovat)

        # p_v = get_p_value(v_0, v_1)
    else:
        normal_dist = False
        # print(len(v_0))
        print('feature can be analized by H test')
        p_v = stats.kruskal(v_0, v_1, v_2)[1]
    if p_v < 0.05:
        print('Feature that has difference among three groups')
        print('p_value<0.05', p_v < 0.05)
        # Multiplt comparison
        print('Doing multicompare')
        if normal_dist:
            # csv_df = multicomp(df['x'], df['y'], feature_name)
            csv_df = sp.posthoc_tukey_hsd(df['x'], df['y'])
        else:
            csv_df = sp.posthoc_dunn(df, val_col='x', group_col='y', p_adjust='holm')
            print('dunn', csv_df)
        return p_v, csv_df, normal_dist
    else:
        print('p_value<0.05', p_v < 0.05)
        return p_v, 0, normal_dist


# histogram of different groups
def plot_hist(feature_1, feature_2, feature_3, title, root_path):
    plt.hist(feature_1, color='g', alpha=0.3, label="label 0", density=1)
    plt.hist(feature_2, color='r', alpha=0.3, label="label 1", density=1)
    plt.hist(feature_3, color='b', alpha=0.3, label="label 2", density=1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(root_path, title+'_hist.png'))
    plt.close()




