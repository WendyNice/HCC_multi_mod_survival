import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import shutil
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math


def create_dir(dir_cre):
    if not os.path.exists(dir_cre):
        os.makedirs(dir_cre)


path = '/data/Wendy/HCC/MR_clf'
path_label = '/data/Wendy/HCC/MR_clf/label_OS_resample'
# path_png_train = os.path.join(path, 'train_img_png')
# path_png_test = os.path.join(path, 'test_img_png')
# path_save_pos = os.path.join(path, 'clf_png/pos')
# path_save_neg = os.path.join(path, 'clf_png/neg')
csv_path = os.path.join(path, 'csv')

test_csv = pd.DataFrame(columns=['path', 'label'])
train_csv = pd.DataFrame(columns=['path', 'label'])
train_val_csv = pd.DataFrame(columns=['path', 'label'])
val_csv = pd.DataFrame(columns=['path', 'label'])
all_csv = pd.DataFrame(columns=['path', 'label'])

#############################################################

train_path = os.path.join(path, 'data_OS_resample', 'train')
data_path_train = os.listdir(train_path)

test_path = os.path.join(path, 'data_OS_resample', 'test')
data_path_test = os.listdir(test_path)

if not os.path.exists(csv_path):
    os.makedirs(csv_path)

train_label = list(np.load(os.path.join(path_label, 'train_label.npy')))
train_id = list(np.load(os.path.join(path_label, 'train_id.npy')))
train_id = [str(i) for i in train_id]

test_label = list(np.load(os.path.join(path_label, 'test_label.npy')))
test_id = list(np.load(os.path.join(path_label, 'test_id.npy')))
test_id = [str(i) for i in test_id]

for i in range(len(train_id)):
    path_new = os.path.join(train_path, train_id[i] + '.nii.gz')
    label_new = train_label[i]
    train_csv.loc[str(i)] = [path_new, label_new]

# train_id, val_id, train_label, val_label = train_test_split(train_id, train_label,
#                                                             test_size=0.3, stratify=train_label)
# for i in range(len(train_id)):
#     path_new = os.path.join(train_path, train_id[i] + '.nii.gz')
#     label_new = train_label[i]
#     train_csv.loc[str(i)] = [path_new, label_new]
#
# for i in range(len(val_id)):
#     path_new = os.path.join(train_path, val_id[i] + '.nii.gz')
#     label_new = val_label[i]
#     val_csv.loc[str(i)] = [path_new, label_new]

for i in range(len(test_id)):
    path_new = os.path.join(test_path, test_id[i] + '.nii.gz')
    label_new = test_label[i]
    test_csv.loc[str(i)] = [path_new, label_new]
print('test_csv', test_csv.shape)
test_csv.to_csv(os.path.join(csv_path, 'val_fold_OS_resample.csv'), header=False, index=False, sep=str(','))
# val_csv.to_csv(os.path.join(csv_path, 'val_fold_OS_resample.csv'), header=False, index=False, sep=str(','))
train_csv.to_csv(os.path.join(csv_path, 'train_fold_OS_resample.csv'), header=False, index=False, sep=str(','))
# train_val_csv.to_csv(os.path.join(csv_path, 'train_val_fold_OS_resample.csv'), header=False, index=False, sep=str(','))
print('train_csv', train_csv.shape, train_csv)
print('test_csv', test_csv.shape, test_csv)
# print('val_csv', val_csv.shape, val_csv)

path_all = '/data/Wendy/HCC/ROI_NII_resample'
pat_all = os.listdir(path_all)
for i in range(len(pat_all)):
    path_new = os.path.join(path_all, pat_all[i])
    # label_new = train_label[i]
    all_csv.loc[str(i)] = [path_new, 1]
all_csv.to_csv(os.path.join(csv_path, 'All_fold_OS_resample.csv'), header=False, index=False, sep=str(','))
print('all_csv', all_csv.shape, all_csv)

