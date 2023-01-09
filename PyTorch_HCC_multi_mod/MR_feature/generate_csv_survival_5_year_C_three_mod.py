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
root_path = '/data/Wendy/HCC/MR_clf'
label_path = '/data/Wendy/HCC/494_8_18.xlsx'
path_data = '/data/Wendy/HCC/ROI_NII'
mod = 'multi_mod'
print('time = 730')
time_set = 730
event_type = '复发时间'

path = os.path.join(root_path, mod)
path_png_train = os.path.join(root_path, mod, 'train_img_png')
path_png_test = os.path.join(root_path, mod, 'test_img_png')
path_save_pos = os.path.join(root_path, mod, 'clf_png/pos')
path_save_neg = os.path.join(root_path, mod, 'clf_png/neg')
csv_path = os.path.join(root_path, mod, 'csv')
#############################################################

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

train_ID = train_data_ori['放射号'].tolist()
test_ID = test_data_ori['放射号'].tolist()
patient_ls = os.listdir(path_data)
for i in train_ID:
    print('i', i)
    if str(i) + '.nii.gz' in patient_ls:
        print('copy')
        shutil.copy(os.path.join(path_data, str(i)+'.nii.gz'), os.path.join(root_path, 'data', 'train'))
# for i in test_ID:
#     print('i', i)
#     if str(i) + '.nii.gz' in patient_ls:
#         shutil.copy(os.path.join(path_data, str(i)+'.nii.gz'), os.path.join(root_path, 'data', 'test'))




