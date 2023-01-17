import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import random
from p_value_2_group import two_independent_sample_test
from p_value_3_group import p_value_3_feature
from datetime import timedelta


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

path_df = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\494_8_18.xlsx'
path_df_valid = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\label_valid.xlsx'
path_os_tr = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\train_df_os.csv '
path_os_ts = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\test_df_os.csv '
path_rfs_tr = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\train_df_rfs.csv '
path_rfs_ts = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\test_df_rfs.csv '

df = pd.read_excel(path_df)
df_val = pd.read_excel(path_df_valid)
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

train_data_df.to_excel(os.path.join(r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\df_os', 'train_data.xlsx'))
test_data_df.to_excel(os.path.join(r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\df_os', 'test_data.xlsx'))
valid_data_df.to_excel(os.path.join(r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\df_os', 'valid_data.xlsx'))

#
#
# # features = ['BCLC', '白蛋白（0>=35,1<35）.1', 'dl_ft']
# features = ['BCLC', '白蛋白（0>=35,1<35）.1']
# # features = ['dl_ft']
# col_event = 'label_event'
# col_time = 'time_event'
# train_data = train_data[features+[col_time, col_event]]
# test_data = test_data[features+[col_time, col_event]]
# valid_data = valid_data[features+[col_time, col_event]]

# x = np.array([
#     (255,0,0,255,1),
#     (255,0,255,255,2),
#     (255,0,255,255,3),
#     (255,255,0,255,2),
#     (255,0,0,255,1),
#     (255,255,255,255,4),
#     ], dtype=[('red', np.ubyte),('green',np.ubyte),('blue',np.ubyte),('alpha',np.ubyte),('width',float)])
# print(x['red'])
# print(x[1]['red'])
# print(x[3])
# num_event_train = np.sum(train_data[col_event])
# num_train = len(train_data[col_event].tolist())
# tr_rate = round(num_event_train/num_train, 4)
# num_event_test = np.sum(test_data[col_event])
# num_test = len(test_data[col_event].tolist())
# ts_rate = round(num_event_test/num_test, 4)
# num_event_valid = np.sum(valid_data[col_event])
# num_valid = len(valid_data[col_event].tolist())
# val_rate = round(num_event_valid/num_valid, 4)

#
# var_1 = train_data['time_event'].tolist()
# var_2 = test_data['time_event'].tolist()
# var_3 = valid_data['time_event'].tolist()
#
# mean_ls = [[change_day_month(mean_fun(var_1)), change_day_month(std_fun(var_1))],
#                [change_day_month(mean_fun(var_2)), change_day_month(std_fun(var_2))],
#                [change_day_month(mean_fun(var_3)), change_day_month(std_fun(var_3))]]
# median_ls = [[change_day_month(median_fun(var_1)), change_day_month(quantile_range(var_1))],
#              [change_day_month(median_fun(var_2)), change_day_month(quantile_range(var_2))],
#              [change_day_month(median_fun(var_3)), change_day_month(quantile_range(var_3))]]
#
# # p_v, csv_df, normal_dist = p_value_3_feature(var_1, var_2, var_3)
# p_v, test_type_select = two_independent_sample_test(var_1, var_3, 0.05)
#
# print('p_v', p_v)
# print('mean_ls', mean_ls)
# print('median_ls', median_ls)



# cph = CoxPHFitter()
# cph = cph.fit(train_data, duration_col=col_time, event_col=col_event)
# cph.print_summary()
# print("*"*50 + 'train_result' + "*"*50)
# print('多因素cox result', cph.score(train_data, scoring_method="concordance_index"))
# print("*"*50 + 'test_result' + "*"*50)
# print('多因素cox result', cph.score(test_data, scoring_method="concordance_index"))
#
# print("*"*50 + 'valid_result' + "*"*50)
# print('多因素cox result', cph.score(valid_data, scoring_method="concordance_index"))
# df_predict_valid = cph.predict_median(valid_data)
# print('df_predict_valid', len(df_predict_valid.tolist()), df_predict_valid.tolist())
