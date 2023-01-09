import pandas as pd
import os
import shutil


def create_dir(dir_cre):
    if not os.path.exists(dir_cre):
        os.makedirs(dir_cre)


target_base = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task003_HCC'
path_data = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task003_HCC_01'
excel_path = '/home/amax/Wendy/SYSU-hcc/table.xlsx'

label_df = pd.read_excel(excel_path)
# label_df = label_df[~label_df['BCLC分期'].isin(['C'])]
print('label_df', label_df)
fangshe_num = [str(int(i)) for i in label_df["放射号"].tolist()]
train_ID = fangshe_num[:int(len(fangshe_num)*0.7)]
test_ID = fangshe_num[int(len(fangshe_num)*0.7):]

join = os.path.join
target_imagesTr = join(target_base, "imagesTr")
target_imagesTs = join(target_base, "imagesTs")
target_labelsTr = join(target_base, "labelsTr")
target_labelsTs = join(target_base, "labelsTs")

create_dir(target_imagesTr)
create_dir(target_imagesTs)
create_dir(target_labelsTr)
create_dir(target_labelsTs)

path_1 = os.path.join(path_data, "imagesTr")
casesList_1 = os.listdir(path_1)
path_2 = os.path.join(path_data, "imagesTs")
casesList_2 = os.listdir(path_2)

casesList = [os.path.join(path_1, i) for i in casesList_1] + [os.path.join(path_2, i) for i in casesList_2]
# print('case', casesList)
for case in casesList:
    print('case', case)
    num = case.split('/')[-1].split('_')[0]
    print('num', num)
    if num in train_ID:
        shutil.copy(case, target_imagesTr)
        if case.endswith('_0000.nii.gz'):
            if 'imagesTr' in case:
                mask_path = case.replace('imagesTr', 'labelsTr').replace('_0000.nii.gz', '.nii.gz')
            else:
                mask_path = case.replace('imagesTs', 'labelsTs').replace('_0000.nii.gz', '.nii.gz')
            shutil.copy(mask_path, target_labelsTr)
    if num in test_ID:
        print('yes')
        shutil.copy(case, target_imagesTs)
        if case.endswith('_0000.nii.gz'):
            if 'imagesTr' in case:
                mask_path = case.replace('imagesTr', 'labelsTr').replace('_0000.nii.gz', '.nii.gz')
            else:
                mask_path = case.replace('imagesTs', 'labelsTs').replace('_0000.nii.gz', '.nii.gz')
            shutil.copy(mask_path, target_labelsTs)


