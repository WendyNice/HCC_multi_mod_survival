import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

path = '/home/amax/Wendy/data/hcc_AP_new/roi_npy'
data_path = os.listdir(path)
patient_ID = [i.split('_')[0] for i in data_path]
csv_path = '/home/amax/Wendy/data/hcc_AP_new/csv'

test_csv = pd.DataFrame(columns=['path', 'label'])

train_csv = pd.DataFrame(columns=['path', 'label'])
train_1_csv = pd.DataFrame(columns=['path', 'label'])
train_2_csv = pd.DataFrame(columns=['path', 'label'])
train_3_csv = pd.DataFrame(columns=['path', 'label'])
train_4_csv = pd.DataFrame(columns=['path', 'label'])
train_5_csv = pd.DataFrame(columns=['path', 'label'])

val_1_csv = pd.DataFrame(columns=['path', 'label'])
val_2_csv = pd.DataFrame(columns=['path', 'label'])
val_3_csv = pd.DataFrame(columns=['path', 'label'])
val_4_csv = pd.DataFrame(columns=['path', 'label'])
val_5_csv = pd.DataFrame(columns=['path', 'label'])

if not os.path.exists(csv_path):
    os.makedirs(csv_path)

excel_path = '/home/amax/Wendy/SYSU-hcc/table.xlsx'
label_df = pd.read_excel(excel_path)
print('label_df', label_df)

fangshe_num = [str(int(i)) for i in label_df["放射号"].tolist()]
print('BCLC分期"].tolist()', label_df["BCLC分期"].tolist())
maiguan = [i for i in label_df["BCLC分期"].tolist()]
print('maiguan', maiguan)

maiguan = [2 if i=="A" else i for i in maiguan]
maiguan = [2 if i=="B" else i for i in maiguan]
maiguan = [1 if i=="C" else i for i in maiguan]
maiguan = [0 if int(i) == 0 else i for i in maiguan]
print('maiguan', maiguan)
label = []
patient_in = []
for i in patient_ID:
    if i in fangshe_num:
        maiguan_num = maiguan[fangshe_num.index(i)]
        if maiguan_num == 1:
            patient_in.append(i)
            label.append(1)
        if maiguan_num == 0:
            # print('np.sum(label)', np.sum(label))
            # if len([j for j in label if j == 0]) <= 96:
                patient_in.append(i)
                label.append(0)
        # print(i, maiguan_num)
print(len(label), len(patient_in))
print(np.sum(label))
X_train_val, X_test, y_train_val, y_test = train_test_split(np.array(patient_in), np.array(label), test_size=0.3, random_state=42, stratify=label)
for i in range(X_train_val.shape[0]):
    path_new = os.path.join(path, X_train_val[i]) + '_3d.npy'
    train_csv.loc[str(i)] = [path_new, y_train_val[i]]

kf = KFold(n_splits=5, random_state=43, shuffle=True)
kf.get_n_splits(X_train_val)
print(kf)
count = 0
for train_index, test_index in kf.split(X_train_val):
    count += 1
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_val = X_train_val[train_index], X_train_val[test_index]
    y_train, y_val = y_train_val[train_index], y_train_val[test_index]
    print('y_train', y_train)

    if count == 1:
        for i in range(X_train.shape[0]):
            path_new = os.path.join(path, X_train[i]) + '_3d.npy'
            train_1_csv.loc[str(i)] = [path_new, y_train[i]]
        for i in range(X_val.shape[0]):
            path_new = os.path.join(path, X_val[i]) + '_3d.npy'
            val_1_csv.loc[str(i)] = [path_new, y_val[i]]
    if count == 2:
        for i in range(X_train.shape[0]):
            path_new = os.path.join(path, X_train[i]) + '_3d.npy'
            train_2_csv.loc[str(i)] = [path_new, y_train[i]]
        for i in range(X_val.shape[0]):
            path_new = os.path.join(path, X_val[i]) + '_3d.npy'
            val_2_csv.loc[str(i)] = [path_new, y_val[i]]
    if count == 3:
        for i in range(X_train.shape[0]):
            path_new = os.path.join(path, X_train[i]) + '_3d.npy'
            train_3_csv.loc[str(i)] = [path_new, y_train[i]]
        for i in range(X_val.shape[0]):
            path_new = os.path.join(path, X_val[i]) + '_3d.npy'
            val_3_csv.loc[str(i)] = [path_new, y_val[i]]
    if count == 4:
        for i in range(X_train.shape[0]):
            path_new = os.path.join(path, X_train[i]) + '_3d.npy'
            train_4_csv.loc[str(i)] = [path_new, y_train[i]]
        for i in range(X_val.shape[0]):
            path_new = os.path.join(path, X_val[i]) + '_3d.npy'
            val_4_csv.loc[str(i)] = [path_new, y_val[i]]
    if count == 5:
        for i in range(X_train.shape[0]):
            path_new = os.path.join(path, X_train[i]) + '_3d.npy'
            train_5_csv.loc[str(i)] = [path_new, y_train[i]]
        for i in range(X_val.shape[0]):
            path_new = os.path.join(path, X_val[i]) + '_3d.npy'
            val_5_csv.loc[str(i)] = [path_new, y_val[i]]


for i in range(len(y_test)):
    path_new = os.path.join(path, X_test[i]) + '_3d.npy'
    print('path_new', path_new, y_test[i])
    test_csv.loc[str(i)] = [path_new, y_test[i]]
print('test_csv', test_csv.shape)
test_csv.to_csv(os.path.join(csv_path, 'test.csv'), header=False, index=False, sep=str(','))
train_csv.to_csv(os.path.join(csv_path, 'train_foldall.csv'), header=False, index=False, sep=str(','))
print('train_csv', train_csv.shape)
train_1_csv.to_csv(os.path.join(csv_path, 'train_fold1.csv'), header=False, index=False, sep=str(','))
train_2_csv.to_csv(os.path.join(csv_path, 'train_fold2.csv'), header=False, index=False, sep=str(','))
train_3_csv.to_csv(os.path.join(csv_path, 'train_fold3.csv'), header=False, index=False, sep=str(','))
train_4_csv.to_csv(os.path.join(csv_path, 'train_fold4.csv'), header=False, index=False, sep=str(','))
train_5_csv.to_csv(os.path.join(csv_path, 'train_fold5.csv'), header=False, index=False, sep=str(','))

val_1_csv.to_csv(os.path.join(csv_path, 'val_fold1.csv'), header=False, index=False, sep=str(','))
val_2_csv.to_csv(os.path.join(csv_path, 'val_fold2.csv'), header=False, index=False, sep=str(','))
val_3_csv.to_csv(os.path.join(csv_path, 'val_fold3.csv'), header=False, index=False, sep=str(','))
val_4_csv.to_csv(os.path.join(csv_path, 'val_fold4.csv'), header=False, index=False, sep=str(','))
val_5_csv.to_csv(os.path.join(csv_path, 'val_fold5.csv'), header=False, index=False, sep=str(','))


print(train_1_csv)


