import pandas as pd
import math
from utils import correlation_analysis, preprocessing_num, pinyin
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt


label_path = '/data/Wendy/HCC/494_8_18.xlsx'
# time_set = 365
time_set = 730
# time_set = 1461
event_type = '复发时间'
# event_type = '死亡时间'

features_in_base = ['BCLC', 'gender', 'age', '肝功能分级', '腹水（有1，无0）', '手术方式（解剖性1，非解剖性0）',
                    'maiguan_invation', '神经侵犯（1.有 2.无 3.不确定）', '周围子灶 ',
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
change_to_date = ['入院日期', '出院日期', '初诊时间', '手术时间', '末次随访时间', '复发时间', '死亡时间']
for i in change_to_date:
    # print(i)
    # for j in data[i].tolist():
    #     print(j)
    #     j_cg = pd.to_datetime(j)
    data[i] = pd.to_datetime(data[i])



print(data[event_type].tolist())
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
data_train_name = name[: int(len(name) * 0.7)]
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

feature_num = ['age', 'INR']

train_data, test_data = preprocessing_num(train_data, test_data, feature_num)
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
train_data = train_data.dropna(axis=0, how='any')
test_data = test_data.dropna(axis=0, how='any')



# 多因素模型
cph = CoxPHFitter(penalizer=1e-2, l1_ratio=1.0)
cph = cph.fit(train_data, duration_col=col_time, event_col=col_event)
cph.print_summary()
summary_df = cph.summary
print('summary_df', summary_df)
p_value = summary_df['p'].tolist()
print('p_value', p_value)
features_in = [features_in[i] for i in range(len(p_value)) if p_value[i] < 0.05]
print('features_in', features_in)
featrues_cox = features_in + [col_event] + [col_time]
train_data = train_data[featrues_cox]
test_data = test_data[featrues_cox]

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