import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


path_img_tr = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task001_HCC/imagesTr'
path_img_ts = '/home/amax/Wendy/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task001_HCC/imagesTs'
png_path = '/home/amax/Wendy/data/hcc_result/png_clf'
save_path = '/home/amax/Wendy/data/hcc_result/image'


path_tr = [os.path.join(path_img_tr, i) for i in os.listdir(path_img_tr)]
path_ts = [os.path.join(path_img_ts, i) for i in os.listdir(path_img_ts)]
path_img = path_tr + path_ts
ID_img = [i.split('/')[-1].split('_')[-2] for i in path_img]
print('path_img', path_img)

path_high = '/home/amax/Wendy/nnUNet/ouput_hcc/high_dice'
file_high = os.listdir(path_high)
print(len(file_high))

path_low = '/home/amax/Wendy/nnUNet/ouput_hcc/low_dice'
file_low = os.listdir(path_low)
print(len(file_low))

path_mask = [os.path.join(path_high, i) for i in file_high]
print(path_mask)

enlarge_size = 5
for mask_path in path_mask:
    print('mask_path', mask_path)
    ID = mask_path.split('/')[-1].split('_')[-1].replace('.nii.gz', '')
    idx = ID_img.index(ID)
    print('path_img', path_img[idx])

    img_mask_nii = sitk.ReadImage(mask_path)
    img_mask_npy = sitk.GetArrayFromImage(img_mask_nii)
    print(img_mask_npy.shape)

    img_nii = sitk.ReadImage(path_img[idx])
    img_npy = sitk.GetArrayFromImage(img_nii)
    print(img_npy.shape)
    # plt.imshow(img_npy[30, :, :])
    # plt.show()

    index_all = []
    lenght_all = 0
    img_roi = img_npy * img_mask_npy
    for idx in range(img_roi.shape[0]):
        sum = np.sum(img_roi[idx, :, :])
        # sum_.append(sum)
        if sum > 0:
            index_all.append(idx)
            lenght_all += 1
    print('index_all', index_all)
    center_loc = index_all[int(len(index_all) / 2)]
    lenght = int(lenght_all / 2) + enlarge_size
    if (lenght > center_loc) or (lenght > img_roi.shape[0] - center_loc):
        lenght = center_loc
        print('lenght', lenght)
        print('lenght larger than center_loc')
    roi_cen = img_roi[center_loc, :, :]
    points_y = np.where(roi_cen > 0)[0]
    points_x = np.where(roi_cen > 0)[1]
    # print('points_x', points_x)
    # print('points_y', points_y)
    center_point_1 = int((np.min(points_y) + np.max(points_y)) / 2)
    lenght_1 = int((np.max(points_y) - np.min(points_y)) / 2) + enlarge_size
    center_point_2 = int((np.min(points_x) + np.max(points_x)) / 2)
    lenght_2 = int((np.max(points_x) - np.min(points_x)) / 2) + enlarge_size
    print('lenght', lenght, lenght_1, lenght_2)

    center_point = [center_loc, center_point_1, center_point_2]
    print('center_point', center_point)
    image_new = np.zeros(img_roi.shape,
                         dtype=img_roi.dtype)

    image_new[center_point[0] - lenght: center_point[0] + lenght,
    center_point[1] - lenght_1: center_point[1] + lenght_1,
    center_point[2] - lenght_2: center_point[2] + lenght_2] = img_npy[
                                                              center_point[0] - lenght: center_point[0] + lenght,
                                                              center_point[1] - lenght_1: center_point[1] + lenght_1,
                                                              center_point[2] - lenght_2: center_point[2] + lenght_2]
    # print(center_point[0]-lenght_1, center_point[0] + lenght_1, center_point[1]-lenght_2, center_point[1] +lenght_2,
    #       center_point[2]-lenght_3, center_point[2] +lenght_3)
    plt.imshow(img_npy[center_loc, :, :], cmap='gray')
    # plt.show()
    plt.savefig(os.path.join(png_path, ID + '_ori.png'))
    plt.close()
    # plt.imshow(image_new[center_loc, :, :], cmap='gray')
    # plt.title(ID)
    # plt.show()
    size = 50
    image_enlarge = np.zeros((img_roi.shape[0] + 2 * size, img_roi.shape[1] + 2 * size, img_roi.shape[2] + 2 * size),
                             dtype=img_roi.dtype)
    # print('image_enlarge', image_enlarge.shape)
    image_enlarge[size:img_roi.shape[0] + size, size:img_roi.shape[1] + size, size:img_roi.shape[2] + size] = image_new
    roi_matrix = image_enlarge[center_point[0]: center_point[0] + 2 * size,
                 center_point[1]: center_point[1] + 2 * size,
                 center_point[2]: center_point[2] + 2 * size]
    plt.imshow(roi_matrix[size, :, :], cmap='gray')
    # plt.show()
    plt.savefig(os.path.join(png_path, ID + '_crop.png'))
    plt.close()
    print('roi_matrix', roi_matrix.shape)
    np.save(os.path.join(save_path, ID + '.npy'), roi_matrix)


