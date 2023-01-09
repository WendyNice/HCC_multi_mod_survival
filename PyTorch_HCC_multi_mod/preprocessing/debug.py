# import os
# import SimpleITK as sitk
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# #
# # # path_img = '/data/Wendy/HCC/resample'
# # path_img = '/data/Wendy/HCC/valid_data/registration_valid'
# # count_regi = 0
# # count_resa = 0
# # for root, dirs, files in os.walk(path_img, topdown=False):
# #     for name in files:
# #         file_name = os.path.join(root, name)
# #         print('file_name', file_name)
# #         if 'registered.nii.gz' in file_name:
# #             count_regi += 1
# #         if 'resam&reor.nii.gz' in file_name:
# #             count_resa += 1
# #
# # print('count_regi', count_regi)
# # print('count_resa', count_resa)
#
# # path = '/data/Wendy/HCC/MR_clf/label'
# # train_label = list(np.load(os.path.join(path, 'train_label.npy')))
# # train_id = list(np.load(os.path.join(path, 'train_id.npy')))
# #
# # test_label = list(np.load(os.path.join(path, 'test_label.npy')))
# # test_id = list(np.load(os.path.join(path, 'test_id.npy')))
#
#
# def slice_select(roi_img):
#     # print('sum', np.sum(roi_img))
#     sum_all = []
#     idx_all = []
#     for idx in range(roi_img.shape[0]):
#         if np.sum(roi_img[idx, :, :]) > 0:
#             # print('idx', idx)
#             sum_all.append(np.sum(roi_img[idx, :, :]))
#             idx_all.append(idx)
#     print('sum_all', sum_all)
#     # print('sum_all.index(', sum_all.index(max(sum_all)))
#     max_idx = sum_all.index(max(sum_all))
#     slice_num = idx_all[max_idx]
#     # print('slice_num, idx_all', slice_num, idx_all)
#     return slice_num, idx_all
#
#
# path_mask = '/data/Wendy/HCC/resample_mask'
# path_AP = [os.path.join(path_mask, i) for i in os.listdir(path_mask) if i.startswith('AP')]
# path_T1WI = [os.path.join(path_mask, i) for i in os.listdir(path_mask) if i.startswith('T1WI')]
#
# id_not_satisfied = []
# for i in range(len(path_AP)):
#     img_AP = sitk.GetArrayFromImage(sitk.ReadImage(path_AP[i]))
#     img_T1WI_path = path_AP[i].replace('AP', 'T1WI')
#     if os.path.exists(img_T1WI_path):
#         img_T1WI = sitk.GetArrayFromImage(sitk.ReadImage(img_T1WI_path))
#         img_cros = img_AP*img_T1WI
#
#         if np.sum(img_cros) == 0:
#             slice_num, idx_all = slice_select(img_AP)
#             roi_AP = img_AP[slice_num, :, :]
#             slice_num, idx_all = slice_select(img_T1WI)
#             roi_T1WI = img_T1WI[slice_num, :, :]
#             plt.imshow(np.concatenate((roi_AP, roi_T1WI), axis=1), cmap='gray')
#             plt.title(path_AP[i])
#             plt.show()
#             print(path_AP[i])
#             id_not_satisfied.append(img_T1WI_path.split('/')[-1].split('_')[-1].replace('.nii.gz', ''))
#         else:
#             print('大于0')
#
# print('id_not_satisfied', id_not_satisfied)

import os
import SimpleITK as sitk
import matplotlib.pyplot as plt


path = '/data/Wendy/HCC/ROI_NII_resample'
pat = os.listdir(path)

for i in pat:
    if '10258579' in i:
        file = os.path.join(path, i)
        img_itk = sitk.ReadImage(file)
        img = sitk.GetArrayFromImage(img_itk)
        print(img.shape)
        img_AP = img[2, :, :, :]
        plt.imshow(img_AP[:, :, 50], cmap='gray')
        plt.show()





