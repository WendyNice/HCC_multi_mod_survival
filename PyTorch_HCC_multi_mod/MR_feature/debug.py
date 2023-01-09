# import nibabel as nib
# import os
# import SimpleITK as sitk
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# pdata_new = '/home/amax/Wendy/SYSU-hcc/registration_2'
# paths = os.listdir(pdata_new)
# print('paths', len(paths))
# path_save = '/home/amax/Wendy/SYSU-hcc/reg_jpg'
# data_ts = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task003_HCC/imagesTs'
# data_tr = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task003_HCC/imagesTr'
#
#
# path_ts = os.listdir(data_ts)
# path_ts = [i.split('_')[0] for i in path_ts if i.endswith("0000.nii.gz")]
# path_tr = os.listdir(data_tr)
# path_tr = [i.split('_')[0] for i in path_tr if i.endswith("0000.nii.gz")]
# path_t = path_ts + path_tr
# print('path_t', len(path_t))
# path_de = [i for i in paths if i not in path_t]
# print('path_de', len(path_de), path_de)
# count = 0
# print('T2WI AP')
# count_fail = 0
# # path_de = ['10283520', '10208231', '10165462', '10242538', '10282641', '10161476']
# for i in paths:
#     path_i = os.path.join(pdata_new, i, 'T2WI', 'T2WI_registered.nii.gz')
#     path_AP = os.path.join(pdata_new, i, 'AP', 'AP_registered.nii.gz')
#     path_PVP = os.path.join(pdata_new, i, 'PVP', 'PVP_registered.nii.gz')
#     path_T1WI = os.path.join(pdata_new, i, 'T1WI', 'T1WI_registered.nii.gz')
#     # path_i = os.path.join(pdata_new, i, 'T2WI',PVP 'data_' + i + '.nii.gz')
#     # path_AP = os.path.join(pdata_new, i, 'AP', 'data_' + i + '.nii.gz')
#     # path_PVP = os.path.join(pdata_new, i, 'PVP', 'data_' + i + '.nii.gz')
#     if os.path.exists(path_i):
#         # pass
#         print('path_i', path_i)
#         img = sitk.ReadImage(path_i)
#         img_npy = sitk.GetArrayFromImage(img)
#         img_AP = sitk.ReadImage(path_AP)
#         img_npy_AP = sitk.GetArrayFromImage(img_AP)
#         img_PVP = sitk.ReadImage(path_PVP)
#         img_npy_PVP = sitk.GetArrayFromImage(img_PVP)
#         img_T1WI = sitk.ReadImage(path_T1WI)
#         img_npy_T1WI = sitk.GetArrayFromImage(img_T1WI)
#
#         # print(img_npy.shape)
#         # print(img_npy_AP.shape)
#         # print(img_npy_PVP.shape)
#         # print(img_npy_T1WI.shape)
#         img_show = img_npy[int(img_npy.shape[0]/2), :, :]
#         img_show_AP = img_npy_AP[int(img_npy_AP.shape[0]/2), :, :]
#         img_show_PVP = img_npy_PVP[int(img_npy_PVP.shape[0] / 2), :, :]
#         img_show_T1WI = img_npy_T1WI[int(img_npy_T1WI.shape[0] / 2), :, :]
#         plt.imshow(np.concatenate((img_show, img_show_AP, img_show_PVP, img_show_T1WI), axis=1), cmap='gray')
#         plt.title(i+'_3')
#         plt.savefig(os.path.join(path_save, i +'.png'))
#         plt.close()
#         # plt.show()
#         count += 1
#         # print(count)
#     else:
#         count_fail += 1
#         print('i', i)
#
# print('count', count)
# print('count_fail', count_fail)
# # pdata = '/home/amax/Wendy/SYSU-hcc/data_sorted'
# # path = os.listdir(pdata)
# # print('path', path)
# #
# # for i in path:
# #     path_i = os.listdir(os.path.join(pdata, i))
# #     path_mod = os.path.join(pdata, i, 'PVP')
# #     file = os.listdir(path_mod)[0]
# #     path_file = os.path.join(path_mod, file)
# #     print('path_file', path_file)
# #     img = sitk.ReadImage(path_file)
# #     img_npy = sitk.GetArrayFromImage(img)
# #     print('img_npy', img_npy.shape)
# #     img_show = img_npy[int(img_npy.shape[0] / 2), :, :]
# #     plt.imshow(img_show, cmap='gray')
# #     plt.show()
#
# # path = '/home/amax/Wendy/SYSU-hcc/registration_2/10161476/AP/data_10161476.nii'
# # img = nib.load(path)

import os
import numpy as np
import pandas as pd
from collections import Counter


path = '/data/Wendy/HCC/MR_clf/csv'
label_list = os.listdir(path)
for i in label_list:
    print(i)
    csv_df = pd.read_csv(os.path.join(path, i))
    # print(i, csv_df)
    label = csv_df.iloc[:, 1].tolist()
    res = Counter(label)
    print(res)

