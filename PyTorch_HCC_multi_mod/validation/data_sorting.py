import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation


def normalization(arr_npy):
    arr_npy = (arr_npy-np.min(arr_npy))/(np.max(arr_npy)-np.min(arr_npy))
    return arr_npy


def new_img(img, radius):
    img_new = np.zeros((img.shape[0] + 2*radius, img.shape[1] + 2*radius, img.shape[2] + 2*radius), dtype=img.dtype)
    img_new[radius:img.shape[0] + radius, radius:img.shape[1] + radius, radius:img.shape[2] + radius] = img
    return img_new


def slice_select(roi_img):
    # print('sum', np.sum(roi_img))
    sum_all = []
    idx_all = []
    for idx in range(roi_img.shape[0]):
        if np.sum(roi_img[idx, :, :]) > 0:
            # print('idx', idx)
            sum_all.append(np.sum(roi_img[idx, :, :]))
            idx_all.append(idx)
    print('sum_all', sum_all)
    # print('sum_all.index(', sum_all.index(max(sum_all)))
    max_idx = sum_all.index(max(sum_all))
    slice_num = idx_all[max_idx]
    # print('slice_num, idx_all', slice_num, idx_all)
    return slice_num, idx_all


def find_center(roi_img_2d):
    points_y = np.where(roi_img_2d > 0)[0]
    points_x = np.where(roi_img_2d > 0)[1]
    point_1 = int((np.min(points_y) + np.max(points_y)) / 2)
    point_2 = int((np.min(points_x) + np.max(points_x)) / 2)
    return point_1, point_2


def roi_gen(path_save_png, name, img, roi, shape, radius):
    # img_new = np.zeros(shape, dtype=img.dtype)
    # print(img.shape, roi.shape, np.sum(img))
    # print('roi', np.sum(roi), np.sum(img_roi))
    # print('sum_roi', np.sum(roi))
    slice_num, idx_all = slice_select(roi)
    for idx in idx_all:
        img_seg_2 = roi[idx, :, :]
        dilated_seg = multi_dilation(img_seg_2, 3)
        # plt.imshow(np.concatenate((dilated_seg, img_seg_2, dilated_seg-img_seg_2), axis=1), cmap='gray')
        # plt.title(name)
        # plt.show()
        roi[idx, :, :] = dilated_seg
    img_roi = img * roi
    roi_cen = img_roi[slice_num, :, :]
    if np.sum(img_roi) == 0:
        print(np.sum(roi[slice_num, :, :]))
        plt.imshow(np.concatenate((roi_cen, img[slice_num, :, :]), axis=1), cmap='gray')
        plt.title(path_img)
        plt.show()
    point_1, point_2 = find_center(roi_cen)

    img_new = img_roi[int(slice_num - radius):int(slice_num + radius),
                                      (point_1 - radius):(point_1 + radius),
                                      (point_2 - radius):(point_2 + radius)]

    plt.imshow(np.concatenate((roi_cen, img[slice_num, :, :]), axis=1), cmap='gray')
    plt.title(path_img)
    # plt.savefig(os.path.join(path_save_png, str(name) + '_ori.png'))
    # plt.close()
    plt.show()
    plt.imshow(np.concatenate((img_new[50, :, :], img_new[:, 50, :], img_new[:, :, 50]), axis=1), cmap='gray')
    plt.title(path_img)
    # plt.savefig(os.path.join(path_save_png, str(name) + '_roi.png'))
    # plt.close()
    plt.show()
    return img_new


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


path_img = '/data/Wendy/HCC/valid_set/resample'
path_save = '/data/Wendy/HCC/valid_set/ROI_NII_resample'
path_save_png = '/data/Wendy/HCC/valid_set/ROI_png_resample'
pat = os.listdir(path_img)


if not os.path.exists(path_save):
    os.makedirs(path_save)
if not os.path.exists(path_save_png):
    os.makedirs(path_save_png)

for i in pat:

    if 10104757 == i:
        print(i)
        # if not os.path.exists(os.path.join(path_save, i + '.nii.gz')):
        # if i != '10208231':
        radius = 50
        img_path = os.path.join(path_img, i)
        AP_path = os.path.join(img_path, 'AP', 'AP_resam&reor.nii.gz')
        PVP_path = os.path.join(img_path, 'PVP', 'PVP_resam&reor.nii.gz')
        T1WI_path = os.path.join(img_path, 'T1WI', 'T1WI_resam&reor.nii.gz')
        # T2WI_path = os.path.join(img_path, 'T2WI', 'T2WI_resam&reor.nii.gz')

        AP_img = sitk.GetArrayFromImage(sitk.ReadImage(AP_path))
        AP_img = normalization(AP_img)
        PVP_img = sitk.GetArrayFromImage(sitk.ReadImage(PVP_path))
        PVP_img = normalization(PVP_img)
        T1WI_img = sitk.GetArrayFromImage(sitk.ReadImage(T1WI_path))
        T1WI_img = normalization(T1WI_img)
        # T2WI_img = sitk.GetArrayFromImage(sitk.ReadImage(T2WI_path))
        # T2WI_img = normalization(T2WI_img)

        AP_mask_path = os.path.join(img_path, 'AP', str(i) + 'AP' + '.nii.gz')
        PVP_mask_path = os.path.join(img_path, 'PVP', str(i) + 'PVP' + '.nii.gz')
        T1WI_mask_path = os.path.join(img_path, 'T1WI', str(i) + 'T1' + '.nii.gz')
        # T2WI_mask_path = os.path.join(path_mask, 'T2WI_mask_' + i + '.nii.gz')

        if os.path.exists(AP_mask_path) and os.path.exists(PVP_mask_path) and os.path.exists(T1WI_mask_path):
            # if os.path.exists(AP_mask_path):
            AP_mask = sitk.GetArrayFromImage(sitk.ReadImage(AP_mask_path))
            PVP_mask = sitk.GetArrayFromImage(sitk.ReadImage(PVP_mask_path))
            T1WI_mask = sitk.GetArrayFromImage(sitk.ReadImage(T1WI_mask_path))
            # T2WI_mask = sitk.GetArrayFromImage(sitk.ReadImage(T2WI_mask_path))

            AP_img = new_img(AP_img, radius)
            PVP_img = new_img(PVP_img, radius)
            T1WI_img = new_img(T1WI_img, radius)
            # T2WI_img = new_img(T2WI_img, radius)

            AP_mask = new_img(AP_mask, radius)
            PVP_mask = new_img(PVP_mask, radius)
            T1WI_mask = new_img(T1WI_mask, radius)
            # T2WI_mask = new_img(T2WI_mask, radius)

            # print('PVP_img', PVP_img.shape, PVP_mask.shape)
            AP_new = roi_gen(path_save_png, i+'AP', AP_img, AP_mask, (100, 100, 100), radius)
            PVP_new = roi_gen(path_save_png, i+'PVP', PVP_img, PVP_mask, (100, 100, 100), radius)
            T1WI_new = roi_gen(path_save_png, i+'T1WI', T1WI_img, T1WI_mask, (100, 100, 100), radius)
            # T2WI_new = roi_gen(path_save_png, i+'T2WI', T2WI_img, T2WI_mask, (80, 80, 80), radius)

            img_new = np.zeros((3, 100, 100, 100), dtype=AP_new.dtype)

            img_new[0, :, :, :] = AP_new
            img_new[1, :, :, :] = PVP_new
            img_new[2, :, :, :] = T1WI_new
            # img_new[3, :, :, :] = T2WI_new
            plt.imshow(np.concatenate((img_new[0, :, :, radius], img_new[1, :, :, radius],
                                       img_new[2, :, :, radius]), axis=1), cmap='gray')
            plt.title(i)
            plt.savefig(os.path.join(path_save_png, str(i) + '_3.png'))
            plt.close()
            nii_img = sitk.GetImageFromArray(img_new)
            sitk.WriteImage(nii_img, os.path.join(path_save, i + '.nii.gz'))





