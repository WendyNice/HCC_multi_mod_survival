import pandas as pd
import math
from utils import correlation_analysis, preprocessing_num, pinyin
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import numpy as np


label_path = '/data/Wendy/HCC/valid_set/label_valid.xlsx'
# train_test_name_path = '/data/Wendy/HCC/MR_clf/train_test_label'
# time_set = 365
time_set = 730
# time_set = 1461
event_type = '复发时间'
# event_type = '死亡时间'

features_in = ['BCLC', '白蛋白（0>=35,1<35）.1', 'AST(1>35;0<=35).1']
data = pd.read_excel(label_path)
data.replace('\\', '', inplace=True)
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
            label_event.append(2)
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


col_event = 'label_event'
col_time = 'time_event'


print('features_in', features_in)

valid_data = data[['放射号'] + features_in + [col_event, col_time]]
valid_data['白蛋白（0>=35,1<35）.1'] = [0 if i >= 35 else 1 for i in valid_data['白蛋白（0>=35,1<35）.1'].tolist()]
valid_data['AST(1>35;0<=35).1'] = [1 if i > 35 else 0 for i in valid_data['AST(1>35;0<=35).1'].tolist()]

path_file = '/data/Wendy/HCC/valid_set/ROI_NII_resample'
pat_all = [int(i.replace('.nii.gz', '')) for i in os.listdir(path_file)]
in_df = valid_data['放射号'].isin(pat_all)
valid_data = valid_data[valid_data['放射号'].isin(pat_all)]
print('valid_data', valid_data)

valid_data.to_csv('/data/Wendy/HCC/valid_set/clinic_feature/valid_RFS.csv')