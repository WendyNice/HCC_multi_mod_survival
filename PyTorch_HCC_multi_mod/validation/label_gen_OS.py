import pandas as pd
import math
from utils import correlation_analysis, preprocessing_num, pinyin
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import numpy as np


label_path = '/data/Wendy/HCC/valid_set/label_valid.xlsx'
# time_set = 365
time_set = 1826
# time_set = 1461
# event_type = '复发时间'
event_type = '死亡时间'

# features_in_base = ['BCLC', 'gender', 'age', '肝功能分级', '腹水（有1，无0）', '手术方式（解剖性1，非解剖性0）',
#                     'maiguan_invation', '神经侵犯（1.有 2.无 3.不确定）', '周围子灶 ',
#                     '胆管侵犯（1.有 2.无 3.不确定）', '血管侵犯1，肝外转移2，无0']
features_in_base = ['BCLC', 'gender', 'age', '肝功能分级', '手术方式（解剖性1，非解剖性0）', '神经侵犯（1.有 2.无 3.不确定）', '周围子灶 ',
                    '胆管侵犯（1.有 2.无 3.不确定）', '血管侵犯1，肝外转移2，无0']

# features_in_blood = ['HBsAg（1.阳性0.阴性）', '肝硬化（1.有 0.无）', 'HBsAg（1.阳性0.阴性）.1', '肝硬化（1.有 0.无）.1',
#                      '白蛋白（0>=35,1<35）', '白蛋白（0>=35,1<35）.1', '总胆红素(2<=34;1>34)', '总胆红素(2<=34;1>34).1',
#                      'INR', 'PT(1>13.5;0<=13.5)', 'PT(1>13.5;0<=13.5).1', 'ALP(1>100;0<=100)', 'ALP(1>100;0<=100).1',
#                      'ALT(1>40;0<=40)', 'ALT(1>40;0<=40).1', 'AST(1>35;0<=35)', 'AST(1>35;0<=35).1', 'GGT(1>50;0<=50)',
#                      'GGT(1>50;0<=50).1', 'PLT(1<100;0>=100)', 'PLT(1<100;0>=100).1',
#                      'AFP(1>200;0<=200).1', 'CA199(1>35,0<=35）', 'CA199(1>35,0<=35）.1', 'CEA(1>5,0<=5)',
#                      'CEA(1>5,0<=5).1', 'FERR(1>400,0<=400)', 'FERR(1>400,0<=400).1']

features_in_blood = ['HBsAg（1.阳性0.阴性）', '肝硬化（1.有 0.无）', 'HBsAg（1.阳性0.阴性）.1',
                     '白蛋白（0>=35,1<35）.1', '总胆红素(2<=34;1>34).1',
                     'INR', 'PT(1>13.5;0<=13.5).1', 'ALP(1>100;0<=100).1',
                     'ALT(1>40;0<=40).1', 'AST(1>35;0<=35).1',
                     'GGT(1>50;0<=50).1', 'PLT(1<100;0>=100).1',
                     'AFP(1>200;0<=200).1',  'CA199(1>35,0<=35）.1',
                     'CEA(1>5,0<=5).1', 'FERR(1>400,0<=400).1']


# features_in_rad = ['BCLC分期', '分级（1.高分化2.中分化3.低分化4.其他（注明））', '包膜侵犯（1.有 2.无 3.不确定）',
#                    '出血（1.有 2.无 3.不确定）', '坏死（1.有 2.无 3.不确定）', '神经侵犯（1.有 2.无 3.不确定）',
#                    '脉管侵犯（1.有 2.无 3.不确定）', '胆管侵犯（1.有 2.无 3.不确定）', '卫星结节（1.有 2.无 3.不确定）',
#                    '数目（1代表单个，2代表2个或以上）', '最大径', '血管侵犯1，肝外转移2，无0', 'nonrim APHE',
#                    'nonperipheral washout', 'enhancing capsule', 'restricted diffusion', '马赛克结构',
#                    '坏死', '出血', '肿块内脂肪', '结中结', '肿瘤边界（光滑0、局部/多重突出/模糊1）',
#                    '包膜（0包膜不完整或无;1包膜完整）', '形态（1不规则，0规则）']


data = pd.read_excel(label_path)
data.replace('\\', '', inplace=True)
change_to_date = ['入院日期', '出院日期', '初诊时间', '手术时间', '末次随访时间', '复发时间', '死亡时间']
for i in change_to_date:
    # print(i)
    print('data[i]', data[i])
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
# data = data[data['label_event'] < 2]

csv_path = '/data/Wendy/HCC/MR_clf/csv'
path_file = '/data/Wendy/HCC/valid_set/ROI_NII_resample'
extra_val_csv = pd.DataFrame(columns=['path', 'label'])
extra_all_val_csv = pd.DataFrame(columns=['path', 'label'])

ID = data['放射号'].tolist()
label = data['label_event'].tolist()
time_event = data['time_event'].tolist()

pat_all = os.listdir(path_file)
for i in range(len(ID)):
    if str(ID[i]) + '.nii.gz' in pat_all:
        if label[i] < 2 or time_event[i] > 1400:
            path_new = os.path.join(path_file, str(ID[i]) + '.nii.gz')
            label_new = label[i]
            extra_val_csv.loc[str(i)] = [path_new, label_new]
        else:
            print('ID[i]', ID[i], label[i], time_event[i])
print('extra_val_csv', extra_val_csv.shape)
extra_val_csv['label'] = extra_val_csv['label'].replace(2, 0)

extra_val_csv.to_csv(os.path.join(csv_path, 'extra_valid_fold_OS_resample.csv'), header=False, index=False, sep=str(','))


for i in range(len(pat_all)):
    # print('pat_all[i]', pat_all[i])
    path_new = os.path.join(path_file, pat_all[i])
    # label_new = train_label[i]
    extra_all_val_csv.loc[str(i)] = [path_new, 1]
extra_all_val_csv.to_csv(os.path.join(csv_path, 'extra_all_valid_fold_OS_resample.csv'), header=False, index=False, sep=str(','))
print('extra_all_val_csv', extra_all_val_csv.shape, extra_all_val_csv)
