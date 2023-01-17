import pandas as pd
from p_value_2_group import two_independent_sample_test
from sklearn.feature_selection import chi2
import numpy as np
from scipy import stats


path_df = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\494_8_18.xlsx'
path_df_valid = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\label_valid.xlsx'

path_os_tr = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\train_df_os.csv '
path_os_ts = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\test_df_os.csv '
path_os_v = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\valid_df_os.csv '

path_rfs_tr = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\train_df_rfs.csv '
path_rfs_ts = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\test_df_rfs.csv '
path_rfs_v = r'D:\Data analysis\Survival_analysis\SYSU_Cancer_HCC\result_sorting\result\valid_df_rfs.csv '

os_tr = pd.read_csv(path_os_tr)
os_ts = pd.read_csv(path_os_ts)
os_v = pd.read_csv(path_os_v)
rfs_tr = pd.read_csv(path_rfs_tr)
rfs_ts = pd.read_csv(path_rfs_ts)
rfs_v = pd.read_csv(path_rfs_v)
id_os_tr = os_tr['ID'].tolist()
id_os_ts = os_ts['ID'].tolist()
id_os_v = os_v['ID'].tolist()
id_rfs_tr = rfs_tr['ID'].tolist()
id_rfs_ts = rfs_ts['ID'].tolist()
id_rfs_v = rfs_v['ID'].tolist()

id_tr_in = []
for i in id_os_tr:
    if i in id_rfs_tr:
        id_tr_in.append(i)
id_ts_in = []
for j in id_os_ts:
    if j in id_rfs_ts:
        id_ts_in.append(j)
id_v_in = []
for j in id_os_v:
    if j in id_rfs_v:
        id_v_in.append(j)

df_basic = pd.read_excel(path_df)
df_basic_valid = pd.read_excel(path_df_valid)
df_basic_valid.replace('\\', np.NaN, inplace=True)
df_basic_valid.replace('/', np.NaN, inplace=True)
df_basic['ID'] = df_basic['放射号']
df_basic_valid['ID'] = df_basic_valid['放射号']
basic_feature = ['BCLC', 'gender', 'age', 'HBsAg（1.阳性0.阴性）.1', '白蛋白（0>=35,1<35）.1', '总胆红素(2<=34;1>34).1',
                  'PT(1>13.5;0<=13.5).1', '肝硬化（1.有 0.无）', 'AFP(1>200;0<=200).1', '肝功能分级',
                 '数目（1代表单个，2代表2个或以上）', '最大径', '血管侵犯1，肝外转移2，无0']
feature_num = ['age', '最大径']

size_mapping = {0: 0, 'A': 1, 'B': 2, 'C': 3}  # 建立一个字典，构建键值对，即数据映射。
df_basic["BCLC"] = df_basic["BCLC"].map(size_mapping)  # map函数的使用
size_mapping = {'A': 0, 'B': 1}  # 建立一个字典，构建键值对，即数据映射。
df_basic["肝功能分级"] = df_basic["肝功能分级"].map(size_mapping)  # map函数的使用

size_mapping = {'A0': 0, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'B': 2, 'c': 3, 'C': 3}  # 建立一个字典，构建键值对，即数据映射。
df_basic_valid["BCLC"] = df_basic_valid["BCLC"].map(size_mapping)  # map函数的使用
size_mapping = {'A': 0, 'B': 1}  # 建立一个字典，构建键值对，即数据映射。
df_basic_valid["肝功能分级"] = df_basic_valid["肝功能分级"].map(size_mapping)  # map函数的使用

df_basic_valid['白蛋白（0>=35,1<35）.1'] = [0 if i >= 35 else 1 for i in df_basic_valid['白蛋白（0>=35,1<35）.1'].tolist()]
df_basic_valid['总胆红素(2<=34;1>34).1'] = [1 if i > 34 else 0 for i in df_basic_valid['总胆红素(2<=34;1>34).1'].tolist()]
df_basic_valid['PT(1>13.5;0<=13.5).1'] = [1 if i > 13 else 0 for i in df_basic_valid['PT(1>13.5;0<=13.5).1'].tolist()]
df_basic_valid['AFP(1>200;0<=200).1'] = [1 if i > 200 else 0 for i in df_basic_valid['AFP(1>200;0<=200).1'].tolist()]


train_df = df_basic[df_basic['ID'].isin(id_tr_in)]
test_df = df_basic[df_basic['ID'].isin(id_ts_in)]
val_df = df_basic_valid[df_basic_valid['ID'].isin(id_v_in)]


event_type = '死亡时间'
change_to_date = ['手术时间', '末次随访时间', '复发时间', '死亡时间']
for i in change_to_date:
    # print(i)
    # for j in data[i].tolist():
    #     print(j)
    #     j_cg = pd.to_datetime(j)
    train_df[i] = pd.to_datetime(train_df[i])
train_df['event_time'] = [i.days for i in train_df[event_type] - train_df['手术时间']]
train_df['follow_time'] = [i.days for i in train_df['末次随访时间'] - train_df['手术时间']]


print()
#
# for feature in basic_feature:
#     train_fea = train_df[feature].tolist()
#     test_fea = val_df[feature].tolist()
#     print('#############################')
#     print('feature', feature)
#     if feature not in feature_num:
#         group = ['train' for i in train_fea] + ['test' for i in test_fea]
#         feature = train_fea + test_fea
#         df = pd.DataFrame(columns=['feature', 'group'])
#         df['feature'] = feature
#         df['group'] = group
#         crosstab = pd.crosstab(df['group'], df['feature'])
#         print('crosstab')
#         print(crosstab)
#         chi2, p_v, dof, ex = stats.chi2_contingency(crosstab)
#     else:
#         p_v, test_type_select = two_independent_sample_test(train_fea, test_fea, 0.05, test_type='two-sided')
#
#     print(p_v)
# print()
#
