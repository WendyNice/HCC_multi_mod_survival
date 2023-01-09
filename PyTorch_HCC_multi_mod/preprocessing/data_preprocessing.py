import copy
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
# import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from skimage.morphology import erosion, dilation


def create_dir(dir_cre):
    if not os.path.exists(dir_cre):
        os.makedirs(dir_cre)


def new_img(img, radius):
    img_new = np.zeros((img.shape[0] + 2*radius, img.shape[1] + 2*radius, img.shape[2] + 2*radius), dtype=img.dtype)
    img_new[radius:img.shape[0] + radius, radius:img.shape[1] + radius, radius:img.shape[2] + radius] = img
    return img_new


def slice_select(roi_img):
    sum_all = []
    idx_all = []
    for idx in range(roi_img.shape[0]):
        if np.sum(roi_img[idx, :, :]) > 0:
            sum_all.append(np.sum(roi_img[idx, :, :]))
            idx_all.append(idx)

    # print('sum_all', sum_all)
    # max_all.append(len(sum_all))
    # print('max_all', max_all)
    # print('max', np.max(max_all))

    max_idx = sum_all.index(max(sum_all))
    slice_num = idx_all[max_idx]
    return slice_num, idx_all


def find_center(roi_img_2d):
    points_y = np.where(roi_img_2d > 0)[0]
    points_x = np.where(roi_img_2d > 0)[1]


    # print('points_x', points_x)
    # print('points_y', points_y)

    point_1 = int((np.min(points_y) + np.max(points_y)) / 2)
    point_2 = int((np.min(points_x) + np.max(points_x)) / 2)
    return point_1, point_2


def multi_dilation(image, iterations):
    kernel_1 = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])
    kernel_2 = np.array([[1, 0, 1],
                         [0, 1, 0],
                         [1, 0, 1]])

    for i in range(iterations):
        image = dilation(image, kernel_1)
        image = dilation(image, kernel_2)
    return image


def process_save(path_img):
    # if '10138951' in path_img:
        print('path_img', path_img)
        img_seg = sitk.ReadImage(os.path.join(gt_path, path_img.replace('_0000.nii.gz', '.nii.gz')))
        img_seg = sitk.GetArrayFromImage(img_seg)
        img_0 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, path_img)))
        img_1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, path_img.replace('_0000.nii.gz', '_0001.nii.gz'))))
        img_2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, path_img.replace('_0000.nii.gz', '_0002.nii.gz'))))
        img_3 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, path_img.replace('_0000.nii.gz', '_0003.nii.gz'))))

        # normalize
        img_0 = (img_0-np.min(img_0))/(np.max(img_0)-np.min(img_0))
        img_1 = (img_1 - np.min(img_1)) / (np.max(img_1) - np.min(img_1))
        img_2 = (img_2-np.min(img_2))/(np.max(img_2)-np.min(img_2))
        img_3 = (img_3-np.min(img_3))/(np.max(img_3)-np.min(img_3))
        # print('img_seg', img_seg.shape)
        # print('img_0', img_0.shape)

        radius = 40
        img_0 = new_img(img_0, radius)
        img_1 = new_img(img_1, radius)
        img_2 = new_img(img_2, radius)
        img_3 = new_img(img_3, radius)
        img_seg = new_img(img_seg, radius)

        slice_num, idx_all = slice_select(img_seg)

        for idx in idx_all:
            img_seg_2 = img_seg[idx, :, :]
            dilated_seg = multi_dilation(img_seg_2, 1)
            # print('dilated_seg', dilated_seg.shape)
            # plt.imshow(np.concatenate((dilated_seg, img_seg_2, dilated_seg-img_seg_2), axis=1), cmap='gray')
            # plt.title(path_img)
            # plt.savefig(os.path.join(path_save_png, path_img.replace('_0000.nii.gz', '') + str(idx) +'.png'))
            # plt.close()
            img_seg[idx, :, :] = dilated_seg

        # print('img_seg', img_seg.shape)
        img_show = copy.deepcopy(img_0)
        img_0 = img_0 * img_seg
        img_1 = img_1 * img_seg
        img_2 = img_2 * img_seg
        img_3 = img_3 * img_seg

        # print('slice_num', slice_num)

        # plt.imshow(np.concatenate((img_0[slice_num, :, :],
        #                            img_3[slice_num, :, :],
        #                            img_show[slice_num, :, :]), axis=1), cmap='gray')
        # plt.title(path_img)
        # plt.savefig(os.path.join(path_save_png, path_img.replace('_0000.nii.gz', '.png')))
        # plt.close()

        img_new = np.zeros((80, 80, 80), dtype=img_0.dtype)
        # print('slice_num-radius/2', int(slice_num-radius/4), int(slice_num+radius/4))

        roi_cen = img_0[slice_num, :, :]
        point_1, point_2 = find_center(roi_cen)
        # print('point', point_1, point_2)
        # print('shape', img_0[int(slice_num-radius/4):int(slice_num+radius/4), (point_1-radius):(point_1+radius),
        #                       (point_2-radius):(point_2+radius)].shape)
        img_new[:int(radius/2), :, :] = img_0[int(slice_num-radius/4):int(slice_num+radius/4), (point_1-radius):(point_1+radius),
                              (point_2-radius):(point_2+radius)]
        img_show = img_show[int(slice_num-radius/4):int(slice_num+radius/4), (point_1-radius):(point_1+radius),
                              (point_2-radius):(point_2+radius)]
        img_new[int(radius/2):radius, :, :] = img_1[int(slice_num - radius/4):int(slice_num + radius/4), (point_1 - radius):(point_1 + radius),
                      (point_2 - radius):(point_2 + radius)]

        img_new[radius:int(radius*3/2), :, :] = img_2[int(slice_num - radius/4):int(slice_num + radius/4), (point_1 - radius):(point_1 + radius),
                                 (point_2 - radius):(point_2 + radius)]
        img_new[int(radius*3/2):radius*2:, :, :] = img_3[int(slice_num - radius/4):int(slice_num + radius/4), (point_1 - radius):(point_1 + radius),
                              (point_2 - radius):(point_2 + radius)]

        plt.figure()
        plt.imshow(np.concatenate((img_new[int(radius / 4), :, :],
                                   img_new[radius + int(radius / 4), :, :],
                                   img_show[int(radius / 4), :, :]), axis=1), cmap='gray')
        plt.title(path_img)
        plt.savefig(os.path.join(path_save_png, path_img.replace('_0000.nii.gz', '_cut.png')))
        plt.close()
        np.save(os.path.join(path_save_npy, path_img.replace('_0000.nii.gz', '.npy')), img_new)
        img_new = sitk.GetImageFromArray(img_new)
        sitk.WriteImage(img_new, os.path.join(path_save, path_img.replace('_0000.nii.gz', '.nii.gz')))

        exe_list.append(path_img)


# # # Train
# # path = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task003_HCC/imagesTr'
# # gt_path = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task003_HCC/labelsTr'
# # path_save = '/home/amax/Wendy/SYSU-hcc/data_clf/multi_mod/Train'
# # path_save_npy = '/home/amax/Wendy/SYSU-hcc/data_clf/multi_mod_npy/Train'
# # path_save_png = '/home/amax/Wendy/SYSU-hcc/data_clf/img_png_train'
#
# # # test
# path = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task003_HCC/imagesTs'
# gt_path = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task003_HCC/labelsTs'
# path_save = '/home/amax/Wendy/SYSU-hcc/data_clf/multi_mod/Test'
# path_save_npy = '/home/amax/Wendy/SYSU-hcc/data_clf/multi_mod_npy/Test'
# path_save_png = '/home/amax/Wendy/SYSU-hcc/data_clf/img_png_test'


# Train
print('80')
path = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task003_HCC/imagesTr'
gt_path = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task003_HCC/labelsTr'
path_save = '/home/amax/Wendy/SYSU-hcc/data_clf/one_mod/Train'
path_save_npy = '/home/amax/Wendy/SYSU-hcc/data_clf/one_mod_npy/Train'
path_save_png = '/home/amax/Wendy/SYSU-hcc/data_clf/img_png_train_one_mod'


exe_list = []

create_dir(path_save)
create_dir(path_save_png)
create_dir(path_save_npy)
paths_img = [i for i in os.listdir(path) if i.endswith('_0000.nii.gz')]
print('len', len(paths_img))
#

for i in paths_img:
    process_save_one_mod(i)

# executor = ThreadPoolExecutor(max_workers=16)
# f_list = []
# for path_img in paths_img:
#     future = executor.submit(process_save, path_img)
#     f_list.append(future)
# print(wait(f_list))

# all_task = executor.map(process_save, paths_img)
# wait(all_task, return_when=ALL_COMPLETED)
print('count', len(exe_list), exe_list)
not_exe = [i for i in paths_img if i not in exe_list]
print('not_exe', not_exe)

# # test
path = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task003_HCC/imagesTs'
gt_path = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task003_HCC/labelsTs'
path_save = '/home/amax/Wendy/SYSU-hcc/data_clf/one_mod/Test'
path_save_npy = '/home/amax/Wendy/SYSU-hcc/data_clf/one_mod_npy/Test'
path_save_png = '/home/amax/Wendy/SYSU-hcc/data_clf/img_png_test_one_mod'

exe_list = []

create_dir(path_save)
create_dir(path_save_png)
create_dir(path_save_npy)
paths_img = [i for i in os.listdir(path) if i.endswith('_0000.nii.gz')]
print('len', len(paths_img))
#

for i in paths_img:
    process_save_one_mod(i)

# executor = ThreadPoolExecutor(max_workers=16)
# f_list = []
# for path_img in paths_img:
#     future = executor.submit(process_save, path_img)
#     f_list.append(future)
# print(wait(f_list))

# all_task = executor.map(process_save, paths_img)
# wait(all_task, return_when=ALL_COMPLETED)
print('count', len(exe_list), exe_list)
not_exe = [i for i in paths_img if i not in exe_list]
print('not_exe', not_exe)
