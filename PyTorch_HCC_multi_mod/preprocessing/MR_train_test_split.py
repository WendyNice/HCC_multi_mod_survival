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
path_data = '/data/Wendy/HCC/ROI_NII_3_mask'
print('time = 730')
time_set = 730
event_type = '复发时间'
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
print('data', data)

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

train_data_ori = data.loc[bool_array_train]
test_data_ori = data.loc[bool_array_test]
print('test_data_ori', test_data_ori)
train_ID = train_data_ori['放射号'].tolist()
test_ID = test_data_ori['放射号'].tolist()
patient_ls = os.listdir(path_data)
print('patient_ls', patient_ls)
train_in = []
train_label = []
test_in = []
test_label = []
del_pat = ['10258812', '10149752']
data_roi_not_match = ['10173145', '10203458', '10152522', '10197879', '10237359', '10263303', '10235302',
                      '10109585', '10108961', '10238775', '10283985', '10262274', '10252492', '10157128',
                      '10134384']

if not os.path.exists(os.path.join(root_path, 'data_3_mask', 'train')):
    os.makedirs(os.path.join(root_path, 'data_3_mask', 'train'))
if not os.path.exists(os.path.join(root_path, 'data_3_mask', 'test')):
    os.makedirs(os.path.join(root_path, 'data_3_mask', 'test'))

for i in range(len(train_ID)):
    if str(int(train_ID[i])) not in del_pat or str(int(train_ID[i])) not in data_roi_not_match:
        # print('train_ID[i]', str(int(train_ID[i])) + '.nii.gz')
        # print(train_data_ori['label_event'].tolist()[i])
        print(str(int(train_ID[i])) + '.nii.gz' in patient_ls)
        if str(int(train_ID[i])) + '.nii.gz' in patient_ls:
            if int(train_data_ori['label_event'].tolist()[i]) < 2:
                print('copy')
                train_in.append(train_ID[i])
                train_label.append(train_data_ori['label_event'].tolist()[i])
                shutil.copy(os.path.join(path_data, str(int(train_ID[i]))+'.nii.gz'), os.path.join(root_path, 'data_3_mask', 'train'))
            else:
                print('label == 2')

for i in range(len(test_ID)):
    print('test_ID', test_ID[i])
    if str(int(test_ID[i])) not in del_pat or str(int(test_ID[i])) not in data_roi_not_match:
        # print('test_ID[i]', test_ID[i])
        if str(int(test_ID[i])) + '.nii.gz' in patient_ls:
            if int(test_data_ori['label_event'].tolist()[i]) < 2:
                print('copy')
                test_in.append(test_ID[i])
                test_label.append(test_data_ori['label_event'].tolist()[i])
                shutil.copy(os.path.join(path_data, str(int(test_ID[i]))+'.nii.gz'), os.path.join(root_path, 'data_3_mask', 'test'))
            else:
                print('label == 2')

np.save(os.path.join(root_path, 'label', 'test_id'), test_in)
np.save(os.path.join(root_path, 'label', 'train_id'), train_in)
np.save(os.path.join(root_path, 'label', 'train_label'), train_label)
np.save(os.path.join(root_path, 'label', 'test_label'), test_label)



