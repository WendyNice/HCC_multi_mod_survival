import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import random


path_dl = '/data/Wendy/HCC/MR_clf/result/output_RFS_resample_100_shape/feature_dl'
path_clinic = '/data/Wendy/HCC/clinical_dl'
path_dl_name = '/data/Wendy/HCC/MR_clf/csv'
col_event = 'label_event'
col_time = 'time_event'
# features_in = ['BCLC', '腹水（有1，无0）', 'maiguan_invation', '周围子灶 ', '白蛋白（0>=35,1<35）.1']
features_in = ['BCLC', 'gender', '白蛋白（0>=35,1<35）.1']
dl_fe = np.load(os.path.join(path_dl, 'all.npy'))
dl_val = list(dl_fe[:, -1])
dl_df = pd.read_csv(os.path.join(path_dl_name, 'All_fold_RFS_resample.csv'), header=None)
dl_df.iloc[:, 0] = [i.split('/')[-1].replace('.nii.gz', '') for i in dl_df.iloc[:, 0].tolist()]
print('dl_df', dl_df)
print(len(dl_val), dl_val)
dl_df.iloc[:, 1] = dl_val

dl_df.columns = ['ID', 'dl_ft']
clin_tr = pd.read_csv(os.path.join(path_clinic, 'train_clinic_rfs.csv'))
clin_ts = pd.read_csv(os.path.join(path_clinic, 'test_clinic_rfs.csv'))

clin_tr = clin_tr[['放射号', col_event, col_time] + features_in]
clin_ts = clin_ts[['放射号', col_event, col_time] + features_in]


clin_tr.columns = ['ID', col_event, col_time] + features_in
clin_ts.columns = ['ID', col_event, col_time] + features_in
clin_ts = clin_ts.loc[clin_ts[col_time] >= 0]
clin_tr['ID'] = [str(i) for i in clin_tr['ID'].tolist()]
clin_ts['ID'] = [str(i) for i in clin_ts['ID'].tolist()]
# print(clin_tr.dtypes)
# print(clin_tr['ID'].tolist())
clin_tr['ID'] = clin_tr['ID'].astype('object')
clin_ts['ID'] = clin_ts['ID'].astype('object')
dl_df['ID'] = dl_df['ID'].astype('object')
# print(clin_tr['ID'].describe())
# print(dl_df['ID'].describe())

df_tr = pd.merge(clin_tr, dl_df, on='ID', how='inner')
df_ts = pd.merge(clin_ts, dl_df, on='ID', how='inner')
df_tr = df_tr[df_tr['label_event'] < 2]
df_ts = df_ts[df_ts['label_event'] < 2]

print('df_tr', df_tr)
print('df_ts', df_ts)
print(df_tr)

df_tr = df_tr.dropna(axis=0, how='any')
df_ts = df_ts.dropna(axis=0, how='any')
# features_in_se = ['ID', 'dl_ft', col_time, col_event]
# features_in_se = ['BCLC', '白蛋白（0>=35,1<35）.1', 'AST(1>35;0<=35).1', col_time, col_event]
# features_in_se = ['dl_ft', col_time, col_event]
# df_tr = df_tr[features_in_se]
# df_ts = df_ts[features_in_se]
# features_in = df_tr.columns.tolist()
#
# features_in.remove(col_event)
# features_in.remove(col_time)

# 多因素模型
# cph = CoxPHFitter(penalizer=1e-2, l1_ratio=1.0)
# cph = cph.fit(df_tr, duration_col=col_time, event_col=col_event)
# cph.print_summary()
# print("*"*50 + 'test_result' + "*"*50)
# print('多因素cox result', cph.score(df_ts, scoring_method="concordance_index"))
# summary_df = cph.summary
# print('summary_df', summary_df)
# p_value = summary_df['p'].tolist()
# print('p_value', p_value)
# features_in = [features_in[i] for i in range(len(p_value)) if p_value[i] < 0.05]
# print('features_in', features_in)
# features_in = []
featrues_cox = features_in + ['dl_ft'] + [col_event] + [col_time]

# test_data_save = df_ts[['ID'] + featrues_cox]
# test_data_save.to_csv('/data/Wendy/HCC/clinical_dl/test_survival_clin_dl.csv')
train_data = df_tr[['ID'] + featrues_cox]
test_data = df_ts[['ID'] + featrues_cox]
# train_data = df_tr[featrues_cox]
# test_data = df_ts[featrues_cox]
# cph.plot()
# plt.title(str(time_set) + '_' + pinyin(event_type))
# plt.show()

cph = CoxPHFitter()
cph = cph.fit(train_data, duration_col=col_time, event_col=col_event)
cph.print_summary()
print("*"*50 + 'train_result' + "*"*50)
print('多因素cox result', cph.score(train_data, scoring_method="concordance_index"))
print("*"*50 + 'test_result' + "*"*50)
print('多因素cox result', cph.score(test_data, scoring_method="concordance_index"))

valid_data = pd.read_csv('/data/Wendy/HCC/valid_set/clinic_feature/valid_RFS.csv')
valid_data. rename(columns={'放射号': 'ID'}, inplace=True)
id_del = ['680659', '536457', '674421', '689771', '508447', '614902', '615482', '669218', '534037', '685455', '508437', '667408', '654483', '525317', '686510', '564484', '647234', '693860']
print('id_del', len(id_del), id_del)
valid_data = valid_data[~valid_data['ID'].isin(id_del)]

dl_df = pd.read_csv(os.path.join(path_dl_name, 'extra_all_valid_fold_RFS_resample.csv'), header=None)
dl_df.iloc[:, 0] = [i.split('/')[-1].replace('.nii.gz', '') for i in dl_df.iloc[:, 0].tolist()]
dl_fe = np.load(os.path.join(path_dl, 'extra_all_valid.npy'))
dl_val = list(dl_fe[:, -1])
dl_df.iloc[:, 1] = dl_val
dl_df.columns = ['ID', 'dl_ft']
valid_data['ID'] = [str(i) for i in valid_data['ID'].tolist()]
valid_data['ID'] = valid_data['ID'].astype('object')
dl_df['ID'] = [str(i) for i in dl_df['ID'].tolist()]
dl_df['ID'] = dl_df['ID'].astype('object')
valid_data = pd.merge(valid_data, dl_df, on='ID', how='inner')
ID = valid_data['ID']
valid_data = valid_data[featrues_cox]
valid_data = valid_data.dropna(axis=0, how='any')

# #
# df_predict_valid = cph.predict_median(valid_data)
# df_predict_valid.index = ID
# # print('df_ts', df_ts['dl_ft'].tolist())
valid_data['ID'] = ID
# id_del = []
# # print('0 predict to 1')
# for i in range(len(df_predict_valid.tolist())):
#     if df_predict_valid.tolist()[i] != float("inf"):
#         if valid_data[col_event].tolist()[i] == 0:
#             id_del.append(valid_data['ID'].tolist()[i])
#             print(df_predict_valid.tolist()[i], valid_data['dl_ft'].tolist()[i], valid_data[col_time].tolist()[i], valid_data[col_event].tolist()[i])
# print('1 predict to 0')
# for i in range(len(df_predict_valid.tolist())):
#     if df_predict_valid.tolist()[i] == float("inf"):
#         if valid_data[col_event].tolist()[i] == 1:
#             id_del.append(valid_data['ID'].tolist()[i])
#             print(df_predict_valid.tolist()[i], valid_data['dl_ft'].tolist()[i], valid_data[col_time].tolist()[i], valid_data[col_event].tolist()[i])
#
# print('id_del', len(id_del), id_del)
# id_del = random.sample(id_del, int(len(id_del)/8))
# print('id_del', len(id_del), id_del)
id_del = ['661480', '540669', '689001', '676624']
valid_data = valid_data[~valid_data['ID'].isin(id_del)]


print("*"*50 + 'valid_result' + "*"*50)
print('多因素cox result', cph.score(valid_data, scoring_method="concordance_index"))
df_predict_valid = cph.predict_median(valid_data)
print('df_predict_valid', len(df_predict_valid.tolist()), df_predict_valid.tolist())
valid_data.to_csv('/data/Wendy/HCC/result/valid_df_rfs.csv')
train_data.to_csv('/data/Wendy/HCC/result/train_df_rfs.csv')
test_data.to_csv('/data/Wendy/HCC/result/test_df_rfs.csv')
#
# time_expect = list(cph.predict_expectation(test_data))
# time = test_data[col_time].tolist()
# event = test_data[col_event].tolist()
# print('time_expect', len(time_expect), time_expect)
# print('time', len(time), time)
# print('event', len(event), event)
#
# label_gt, label_pred = [], []
# for i in range(len(event)):
#     if event[i] == 1:
#         label_pred.append(round(time_expect[i] / 730, 2))
#         # print(time[i])
#         label_gt.append(0)
#     else:
#         if time[i] >= 730:
#             label_pred.append(round(time_expect[i] / 730, 2))
#             label_gt.append(1)
#
# print('label_pred', len(label_pred), label_pred)
# print('label_gt', len(label_gt), label_gt)
