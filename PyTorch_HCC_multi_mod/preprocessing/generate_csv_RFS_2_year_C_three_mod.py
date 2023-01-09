import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import shutil
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math


def get_data(patient_ID, fangshe_num, feature):
    label = []
    patient_in = []
    for id in patient_ID:
        if str(int(id)) in fangshe_num:
            feature_num = feature[fangshe_num.index(str(int(id)))]
            # path_npy = os.path.join(train_path, id + '.npy')
            # img = np.load(path_npy)
            # img_mid = img[10, :, :]
            if feature_num == 1:
                patient_in.append(id)
                # plt.imshow(img_mid, cmap='gray')
                # plt.savefig(os.path.join('/home/amax/Wendy/Kaggle/brain_tumor_radiogenomic_clf/MGMT/MGMT_pos', id + '.png'))
                # plt.close()
                label.append(1)
            if feature_num == 0:
                # print('np.sum(label)', np.sum(label))
                # if len([j for j in label if j == 0]) <= 96:
                    patient_in.append(id)
                    # plt.imshow(img_mid, cmap='gray')
                    # plt.savefig(os.path.join('/home/amax/Wendy/Kaggle/brain_tumor_radiogenomic_clf/MGMT/MGMT_neg', id + '.png'))
                    # plt.close()
                    label.append(0)
    return patient_in, label


def create_dir(dir_cre):
    if not os.path.exists(dir_cre):
        os.makedirs(dir_cre)


#############################################################
mod = 'multi_mod'
path = os.path.join('/home/amax/Wendy/SYSU-hcc/data_clf_2', mod)
path_png_train = os.path.join('/home/amax/Wendy/SYSU-hcc/data_clf_2', mod, 'train_img_png')
path_png_test = os.path.join('/home/amax/Wendy/SYSU-hcc/data_clf_2', mod, 'test_img_png')
path_save_pos = os.path.join('/home/amax/Wendy/SYSU-hcc/data_clf_2', mod, 'clf_png/pos')
path_save_neg = os.path.join('/home/amax/Wendy/SYSU-hcc/data_clf_2', mod, 'clf_png/neg')
csv_path = os.path.join('/home/amax/Wendy/SYSU-hcc/data_clf_2', mod, 'csv')
#############################################################

# train
train_path = os.path.join(path, 'train_npy')
data_path_train = os.listdir(train_path)
print('data_path_train', data_path_train)
patient_ID_train = [i.split('.npy')[0] for i in data_path_train if i.endswith('.npy')]
print('patient_ID_train', len(patient_ID_train), patient_ID_train)

test_path = os.path.join(path, 'test_npy')
data_path_test = os.listdir(test_path)
print('data_path_test', len(data_path_test), data_path_test)
patient_ID_test = [i.split('.npy')[0] for i in data_path_test if i.endswith('.npy')]
print('patient_ID_test', len(patient_ID_test), patient_ID_test)

print('one')


test_csv = pd.DataFrame(columns=['path', 'label'])

train_csv = pd.DataFrame(columns=['path', 'label'])

val_csv = pd.DataFrame(columns=['path', 'label'])
if not os.path.exists(csv_path):
    os.makedirs(csv_path)

excel_path = '/home/amax/Wendy/SYSU-hcc/table.xlsx'
label_df = pd.read_excel(excel_path)
#############################################################
# label_df = label_df[~label_df['BCLC分期'].isin(['C'])]
print('label_df', label_df)
#############################################################

fangshe_num = [str(int(i)) for i in label_df["放射号"].tolist()]
print('fangshe_num', fangshe_num)

change_to_date = ['入院日期', '出院日期', '初诊时间', '手术时间', '末次随访时间', '复发时间', '死亡时间']
for i in change_to_date:
    print(i)
    # for j in data[i].tolist():
    #     print(j)
    #     j_cg = pd.to_datetime(j)
    label_df[i] = pd.to_datetime(label_df[i])

print('time = 730')
time_set = 730
# time_set = 730
print('复发时间', label_df['复发时间'].tolist())
print('手术时间', label_df['手术时间'].tolist())
label_df['event_time'] = [i.days for i in label_df['复发时间'] - label_df['手术时间']]
label_df['follow_time'] = [i.days for i in label_df['末次随访时间'] - label_df['手术时间']]
print('time', label_df['event_time'].tolist())
print('follow_time', label_df['follow_time'].tolist())
# classification label
label_event = []
time_event = []
label_follow = []
time_follow = []
for i in range(len(label_df['event_time'].tolist())):
    if not math.isnan(label_df['event_time'].tolist()[i]):
        label_follow.append(1)
        time_follow.append(label_df['event_time'].tolist()[i])
        if label_df['event_time'].tolist()[i] > time_set:
            label_event.append(0)
            time_event.append(time_set)
        else:
            label_event.append(1)
            time_event.append(label_df['event_time'].tolist()[i])
    else:
        label_follow.append(0)
        time_follow.append(label_df['follow_time'].tolist()[i])
        if label_df['follow_time'].tolist()[i] > time_set:
            label_event.append(0)
            time_event.append(time_set)
        else:
            label_event.append(2)
            time_event.append(label_df['follow_time'].tolist()[i])
label_df['label_event'] = label_event
label_df['time_event'] = time_event
label_df['label_follow'] = label_follow
label_df['time_follow'] = time_follow
print('data', label_df.columns.tolist())
print('label 1', len([i for i in label_df['label_event'].tolist() if i == 1]))
print('label 0', len([i for i in label_df['label_event'].tolist() if i == 0]))
# print('MGMT_value', label_df["MGMT_value"].tolist())
# maiguan = label_df["脉管侵犯（1.有 2.无 3.不确定）"].tolist()
print('label_event', label_event)
feature_event = label_df['label_event'].tolist()
print('feature_event', feature_event)

#############################################################
patient_in, label = get_data(patient_ID_train, fangshe_num, feature_event)
#############################################################

print('patient_in', patient_in)
print('label', label)

create_dir(path_save_neg)
create_dir(path_save_pos)
for idx in range(len(patient_in)):
    path_new = os.path.join(path_png_train, patient_in[idx]) + '_cut.png'
    if os.path.exists(path_new):
        if label[idx] == 0:
            shutil.copy(path_new, path_save_neg)
        else:
            shutil.copy(path_new, path_save_pos)


print('lenght', len(label), len(patient_in))
print(np.sum(label))
for i in range(len(patient_in)):
    path_new = os.path.join(train_path, patient_in[i]) + '.npy'
    # print('path_new', path_new)
    train_csv.loc[str(i)] = [path_new, label[i]]

# train_patient_in, val_patient_in, label_train, label_val = train_test_split(patient_in, label, test_size=0.3, stratify=label)
# print('label_train', len(label_train), len(label_val))
# print(np.sum(label_train), np.sum(label_val))
# for i in range(len(train_patient_in)):
#     path_new = os.path.join(train_path, train_patient_in[i]) + '.npy'
#     # print('path_new', path_new)
#     train_csv.loc[str(i)] = [path_new, label_train[i]]
# for i in range(len(val_patient_in)):
#     path_new = os.path.join(train_path, val_patient_in[i]) + '.npy'
#     # print('path_new', path_new)
#     val_csv.loc[str(i)] = [path_new, label_val[i]]
#############################################################
patient_in_ts, label_ts = get_data(patient_ID_test, fangshe_num, feature_event)
#############################################################

print('lenght', len(label_ts), len(patient_in_ts))
print(np.sum(label_ts))

for i in range(len(label_ts)):
    path_new = os.path.join(test_path, patient_in_ts[i]) + '.npy'
    # print('path_new', path_new, label_ts[i])
    test_csv.loc[str(i)] = [path_new, label_ts[i]]
print('test_csv', test_csv.shape)
test_csv.to_csv(os.path.join(csv_path, 'val_fold3mod_2_RFS.csv'), header=False, index=False, sep=str(','))
train_csv.to_csv(os.path.join(csv_path, 'train_fold3mod_2_RFS.csv'), header=False, index=False, sep=str(','))
# val_csv.to_csv(os.path.join(csv_path, 'val_fold1.csv'), header=False, index=False, sep=str(','))
print('train_csv', train_csv.shape)
print('test_csv', test_csv.shape)
print('train_csv', train_csv)
print('test_csv', test_csv)


