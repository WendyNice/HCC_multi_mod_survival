import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import random
from p_value_2_group import two_independent_sample_test
from p_value_3_group import p_value_3_feature
from datetime import datetime


def mean_fun(val):
    val = np.round(np.mean(val), decimals=2)
    return val


def std_fun(val):
    val = np.round(np.std(val), decimals=2)
    return val


def median_fun(val):
    val = np.round(np.median(val), decimals=2)
    return val


def quantile_range(val):
    val_low = np.round(np.quantile(val, 0.25, interpolation='lower'), decimals=2)
    val_high = np.round(np.quantile(val, 0.75, interpolation='lower'), decimals=2)
    print('val', val)
    print('val_low', val_low, val_high)
    val = val_high-val_low
    return val


def change_day_month(day_num):
    years = day_num // 365
    months = years * 12
    months += (day_num - years * 365) / 30
    return round(months, 0)


def get_month(begin, end):
    begin_year, end_year = begin.year, end.year
    begin_month, end_month = begin.month, end.month
    if begin_month == end_year:
        months = end_month-begin_month
    else:
        months = (end_year-begin_year)*12+end_month-begin_month
    return months


path_df = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\494_8_18.xlsx'
path_df_valid = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\label_valid.xlsx'
path_os_tr = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\train_df_os.csv '
path_os_ts = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\test_df_os.csv '
path_rfs_tr = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\train_df_rfs.csv '
path_rfs_ts = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\test_df_rfs.csv '
event_type = '复发时间'
df = pd.read_excel(path_df)
change_to_date = ['手术时间', '末次随访时间', '复发时间', '死亡时间']
for i in change_to_date:
    # print(i)
    # for j in data[i].tolist():
    #     print(j)
    #     j_cg = pd.to_datetime(j)
    time_ls_new = []
    time_ls = df[i].tolist()
    for j in time_ls:
        j_str = str(j)
        print(type(j_str), j_str, len(j_str))
        if ('-' in j_str) or (j_str == 'NaT'):
            time_ls_new.append(j)
        else:
            time_ls_new.append(datetime.strptime(j, '%Y-%m-%d %H:%M:%S'))
    df[i] = time_ls_new

print('df', df)

ls_begin = df['手术时间'].tolist()
ls_end = df['末次随访时间'].tolist()
ls_event = []
for idx in range(len(ls_end)):
    if ('-' in str(ls_end[idx])) or (ls_end[idx] == 'NaT'):
        ls_event.append(get_month(ls_begin[idx], ls_end[idx]))
    else:
        ls_event.append(0)
df['follow_time_month'] = ls_event

df_val = pd.read_excel(path_df_valid)
df_val.replace('\\', np.NaN, inplace=True)
df_val.replace('/', np.NaN, inplace=True)
change_to_date = ['手术时间', '末次随访时间', '复发时间', '死亡时间']
for i in change_to_date:
    # print(i)
    # for j in data[i].tolist():
    #     print(j)
    #     j_cg = pd.to_datetime(j)
    time_ls_new = []
    time_ls = df_val[i].tolist()

    for j in time_ls:
        j_str = str(j)
        print(type(j_str), j_str, len(j_str))
        if ('-' in j_str) or (j_str == 'nan'):
            time_ls_new.append(j)
        else:
            print(j)
            time_ls_new.append(datetime.strptime(j, '%Y-%m-%d %H:%M:%S'))
    df_val[i] = time_ls_new
print('df_val', df_val)
ls_begin = df_val['手术时间'].tolist()
ls_end = df_val['末次随访时间'].tolist()
ls_event = []
for idx in range(len(ls_end)):
    if ('-' in str(ls_end[idx])) or (ls_end[idx] == 'NaT'):
        ls_event.append(get_month(ls_begin[idx], ls_end[idx]))
    else:
        ls_event.append(0)
df_val['follow_time_month'] = ls_event

os_tr = pd.read_csv(path_os_tr)
os_ts = pd.read_csv(path_os_ts)
rfs_tr = pd.read_csv(path_rfs_tr)
rfs_ts = pd.read_csv(path_rfs_ts)
id_os_tr = os_tr['ID'].tolist()
id_os_ts = os_ts['ID'].tolist()
id_rfs_tr = rfs_tr['ID'].tolist()
id_rfs_ts = rfs_ts['ID'].tolist()

id_tr_in = []
for i in id_os_tr:
    if i in id_rfs_tr:
        id_tr_in.append(i)
id_ts_in = []
for j in id_os_ts:
    if j in id_rfs_ts:
        id_ts_in.append(j)

path = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result'
train_data = pd.read_csv(os.path.join(path, 'train_df_os.csv'))
test_data = pd.read_csv(os.path.join(path, 'test_df_os.csv'))
train_data = train_data[train_data['ID'].isin(id_tr_in)]
test_data = test_data[test_data['ID'].isin(id_ts_in)]
valid_data = pd.read_csv(os.path.join(path, 'valid_df_os.csv'))
train_data_df = pd.merge(train_data, df, how='inner', left_on='ID', right_on='放射号')
test_data_df = pd.merge(test_data, df, how='inner', left_on='ID', right_on='放射号')
valid_data_df = pd.merge(valid_data, df_val, how='inner', left_on='ID', right_on='放射号')

var_1 = train_data_df['follow_time_month'].tolist()
var_2 = test_data_df['follow_time_month'].tolist()
var_3 = valid_data_df['follow_time_month'].tolist()

mean_ls = [[mean_fun(var_1), std_fun(var_1)],
               [mean_fun(var_2), std_fun(var_2)],
               [mean_fun(var_3), std_fun(var_3)]]
median_ls = [[median_fun(var_1), quantile_range(var_1)],
             [median_fun(var_2), quantile_range(var_2)],
             [median_fun(var_3), quantile_range(var_3)]]

# p_v, csv_df, normal_dist = p_value_3_feature(var_1, var_2, var_3)
p_v, test_type_select = two_independent_sample_test(var_1, var_3, 0.05)
p_v_2, test_type_select_2 = two_independent_sample_test(var_1, var_2, 0.05)

print('p_v', p_v)
print('p_v_2', p_v_2)

print('mean_ls', mean_ls)
print('median_ls', median_ls)


