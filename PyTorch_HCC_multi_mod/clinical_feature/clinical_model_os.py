import pandas as pd
import math
from utils import correlation_analysis, preprocessing_num, pinyin
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import numpy as np


label_path = '/data/Wendy/HCC/494_8_18.xlsx'
train_test_name_path = '/data/Wendy/HCC/MR_clf/train_test_label'
# time_set = 365
time_set = 1826
# time_set = 1461
# event_type = '复发时间'
event_type = '死亡时间'


# features_in_base = ['BCLC', 'gender', '肝功能分级', '手术方式（解剖性1，非解剖性0）']
# features_in_blood = ['肝硬化（1.有 0.无）', 'HBsAg（1.阳性0.阴性）.1',
#                      '白蛋白（0>=35,1<35）.1', '总胆红素(2<=34;1>34).1',
#                      'PT(1>13.5;0<=13.5).1', 'ALP(1>100;0<=100).1',
#                      'ALT(1>40;0<=40).1', 'AST(1>35;0<=35).1',
#                      'GGT(1>50;0<=50).1', 'PLT(1<100;0>=100).1',
#                      'AFP(1>200;0<=200).1',  'CA199(1>35,0<=35）.1', 'CEA(1>5,0<=5).1']

features_in_base = ['BCLC', 'gender', 'age']
features_in_blood = ['白蛋白（0>=35,1<35）.1', '总胆红素(2<=34;1>34).1',
                     'AFP(1>200;0<=200).1']

# features_in_base = []
# features_in_blood = ['包膜侵犯（1.有 2.无 3.不确定）', '出血（1.有 2.无 3.不确定）', '坏死（1.有 2.无 3.不确定）', '神经侵犯（1.有 2.无 3.不确定）',
#                    '脉管侵犯（1.有 2.无 3.不确定）', '胆管侵犯（1.有 2.无 3.不确定）', '卫星结节（1.有 2.无 3.不确定）',
#                    '数目（1代表单个，2代表2个或以上）', '最大径', '血管侵犯1，肝外转移2，无0', 'nonrim APHE',
#                    'nonperipheral washout', 'enhancing capsule', 'restricted diffusion', '马赛克结构',
#                    '坏死', '出血', '肿块内脂肪', '结中结', '肿瘤边界（光滑0、局部/多重突出/模糊1）',
#                    '包膜（0包膜不完整或无;1包膜完整）', '形态（1不规则，0规则）']

data = pd.read_excel(label_path)
change_to_date = ['入院日期', '出院日期', '初诊时间', '手术时间', '末次随访时间', '复发时间', '死亡时间']
for i in change_to_date:
    # print(i)
    # for j in data[i].tolist():
    #     print(j)
    #     j_cg = pd.to_datetime(j)
    data[i] = pd.to_datetime(data[i])

print('data', data)

data['event_time'] = [i.days for i in data[event_type] - data['手术时间']]
data['follow_time'] = [i.days for i in data['末次随访时间'] - data['手术时间']]
print('time', data['event_time'].tolist())
print('follow_time', data['follow_time'].tolist())
# classification label
label_event = []
time_event = []
label_follow = []
time_follow = []
for i in range(len(data['event_time'].tolist())):
    if not math.isnan(data['event_time'].tolist()[i]):
        label_follow.append(1)
        time_follow.append(data['event_time'].tolist()[i])
        if data['event_time'].tolist()[i] > time_set:
            label_event.append(0)
            time_event.append(time_set)
        else:
            label_event.append(1)
            time_event.append(data['event_time'].tolist()[i])
    else:
        label_follow.append(0)
        time_follow.append(data['follow_time'].tolist()[i])
        if data['follow_time'].tolist()[i] > time_set:
            label_event.append(0)
            time_event.append(time_set)
        else:
            label_event.append(0)
            time_event.append(data['follow_time'].tolist()[i])
data['label_event'] = label_event
data['time_event'] = time_event
data['label_follow'] = label_follow
data['time_follow'] = time_follow
print('data', data.columns.tolist())

size_mapping = {0: 0, 'A': 1, 'B': 2, 'C': 3}  # 建立一个字典，构建键值对，即数据映射。
data["BCLC"] = data["BCLC"].map(size_mapping)  # map函数的使用
size_mapping = {'A': 0, 'B': 1}  # 建立一个字典，构建键值对，即数据映射。
data["肝功能分级"] = data["肝功能分级"].map(size_mapping)  # map函数的使用

# train test split
name = data['放射号'].tolist()
# data_train_name, data_test_name = train_test_split(name, test_size=0.3, random_state=42)
data_train_name = list(np.load('/data/Wendy/HCC/MR_clf/train_test_label/train_name_clinic.npy'))
data_test_name = list(np.load('/data/Wendy/HCC/MR_clf/train_test_label/test_name_clinic.npy'))
# np.save(os.path.join(train_test_name_path, 'train_name_clinic.npy'), np.array(data_train_name))
# np.save(os.path.join(train_test_name_path, 'test_name_clinic.npy'), np.array(data_test_name))
# data_train_name = name[: int(len(name) * 0.7)]
print('data_train_name', data_train_name)
bool_array_train = []
bool_array_test = []
for i in data['放射号'].tolist():
    if i in data_train_name:
        bool_array_train.append(True)
        bool_array_test.append(False)
    else:
        bool_array_train.append(False)
        bool_array_test.append(True)
thresh_p = 0.0001
train_data_ori = data.loc[bool_array_train]
test_data_ori = data.loc[bool_array_test]
features_in = features_in_base + features_in_blood
correlation_threshold = 0.9
train_data = train_data_ori[features_in]
to_drop = correlation_analysis(train_data, correlation_threshold)
print('to_drop', to_drop)
features_in = [i for i in features_in if i not in to_drop]

train_data = train_data[features_in]
test_data = test_data_ori[features_in]

col_event = 'label_event'
col_time = 'time_event'

print('features_in', features_in)
# features_in = [pinyin(i) for i in features_in]
# train_data.columns = features_in
# test_data.columns = features_in
train_data[col_event] = train_data_ori[col_event]
train_data[col_time] = train_data_ori[col_time]
test_data[col_event] = test_data_ori[col_event]
test_data[col_time] = test_data_ori[col_time]
print(train_data.isnull().sum())
print(test_data.isnull().sum())

train_data_ori.to_csv(os.path.join('/data/Wendy/HCC/clinical_dl', 'train_clinic.csv'))
test_data_ori.to_csv(os.path.join('/data/Wendy/HCC/clinical_dl', 'test_clinic.csv'))
train_data = train_data.dropna(axis=0, how='any')
test_data = test_data.dropna(axis=0, how='any')
train_data.to_csv('/data/Wendy/HCC/result/multi_variants/train_df_os.csv')
test_data.to_csv('/data/Wendy/HCC/result/multi_variants/test_df_os.csv')

# 多因素模型
cph = CoxPHFitter(penalizer=0.001, l1_ratio=1)
cph = cph.fit(train_data, duration_col=col_time, event_col=col_event)
cph.print_summary()
summary_df = cph.summary
print('summary_df', summary_df)
p_value = summary_df['p'].tolist()
print('p_value', p_value)
features_in = [features_in[i] for i in range(len(p_value)) if p_value[i] < 0.05]
print('features_in', features_in)
# features_in = ['BCLC', 'AFP(1>200;0<=200).1']
featrues_cox = features_in + [col_event] + [col_time]
train_data = train_data[featrues_cox]
test_data = test_data[featrues_cox]
test_data = test_data.loc[test_data[col_time] >= 0]
# cph.plot()
# plt.title(str(time_set) + '_' + pinyin(event_type))
# plt.show()
print('train_data', train_data)
cph = CoxPHFitter()
cph = cph.fit(train_data, duration_col=col_time, event_col=col_event)
cph.print_summary()
print("*"*50 + 'train_result' + "*"*50)
print('多因素cox result', cph.score(train_data, scoring_method="concordance_index"))
print("*"*50 + 'test_result' + "*"*50)
print('多因素cox result', cph.score(test_data, scoring_method="concordance_index"))

valid_data = pd.read_csv('/data/Wendy/HCC/valid_set/clinic_feature/valid_OS.csv')

valid_data. rename(columns={'放射号': 'ID'}, inplace=True)
id_del = ['680659', '536457', '674421', '689771', '508447', '614902', '615482', '669218', '534037', '685455', '508437', '667408', '654483', '525317', '686510', '564484', '647234', '693860']
valid_data = valid_data[~valid_data['ID'].isin(id_del)]
valid_data = valid_data[valid_data['label_event'] < 2]

valid_data = valid_data[featrues_cox]
valid_data = valid_data.dropna(axis=0, how='any')


# print('valid_data', valid_data)
print("*"*50 + 'valid_result' + "*"*50)
print('多因素cox result', cph.score(valid_data, scoring_method="concordance_index"))
df_predict_valid = cph.predict_median(valid_data)
df_predict_test = cph.predict_median(test_data)
print('df_predict_test', df_predict_test.tolist())
print('df_predict_valid', df_predict_valid.tolist())



