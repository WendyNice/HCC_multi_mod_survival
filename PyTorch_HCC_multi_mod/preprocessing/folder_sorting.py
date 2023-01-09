import os
import shutil


def move_file(src_dir, tar_dir):
    if not os.path.exists(tar_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(tar_dir)
    shutil.copy(src_dir, tar_dir)


path = '/data/Wendy/HCC/HCC'
path_output = '/data/Wendy/HCC/data_sort'

AP = 'AP'
PVP = 'PVP'
T1WI = 'T1WI'
T2WI = 'T2WI'

path_AP = os.path.join(path, AP)

# img startwith data, mask startwith mask
patients_AP = []
patient_ls = os.listdir(path_AP)
for j in patient_ls:
    # print('path_AP', path_AP)
    # print(i)
    # print(j)
    file_p = os.path.join(path_AP, j)
    print('file_p', file_p)
    src_file = [os.path.join(file_p, o) for o in os.listdir(file_p) if o.startswith('data')][0]

    patients_AP.append(src_file)

    tar_file = os.path.join(path_output, j, 'AP')
    print('tar_file', tar_file)
    if len(src_file) > 0:
        move_file(src_file, tar_file)
print('patients_PVP', len(patients_AP), patients_AP)


# img startwith data, mask startwith mask
path_PVP = os.path.join(path, PVP)
#
patients_PVP = []
patient_ls = os.listdir(path_PVP)
for j in patient_ls:
    # print('path_AP', path_AP)
    # print(i)
    # print(j)
    file_p = os.path.join(path_PVP, j)
    print('file_p', file_p)
    src_file = [os.path.join(file_p, o) for o in os.listdir(file_p) if o.startswith('data')][0]

    patients_PVP.append(src_file)

    tar_file = os.path.join(path_output, j, 'PVP')
    print('tar_file', tar_file)
    if len(src_file) > 0:
        move_file(src_file, tar_file)
print('patients_PVP', len(patients_PVP), patients_PVP)

path_T1WI = os.path.join(path, T1WI)
patients_T1WI = []
patient_ls = os.listdir(path_T1WI)
for j in patient_ls:
    # print('path_AP', path_AP)
    # print(i)
    # print(j)
    file_p = os.path.join(path_T1WI, j)
    print('file_p', file_p)
    src_file = [os.path.join(file_p, o) for o in os.listdir(file_p) if o.startswith('data')][0]

    patients_T1WI.append(src_file)

    tar_file = os.path.join(path_output, j, 'T1WI')
    print('tar_file', tar_file)
    if len(src_file) > 0:
        move_file(src_file, tar_file)
print('patients_T1WI', len(patients_T1WI), patients_T1WI)

path_T2WI = os.path.join(path, T2WI)
patients_T2WI = []
patient_ls = os.listdir(path_T2WI)
for j in patient_ls:
    # print('path_AP', path_AP)
    # print(i)
    # print(j)
    file_p = os.path.join(path_T2WI, j)
    print('file_p', file_p)
    src_file = [os.path.join(file_p, o) for o in os.listdir(file_p) if o.startswith('data')][0]

    patients_T2WI.append(src_file)

    tar_file = os.path.join(path_output, j, 'T2WI')
    print('tar_file', tar_file)
    if len(src_file) > 0:
        move_file(src_file, tar_file)
print('patients_T2WI', len(patients_T2WI), patients_T2WI)

