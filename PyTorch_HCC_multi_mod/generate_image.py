import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def dcm_reader(patient_path):
    reader = sitk.ImageSeriesReader()
    slice_names = reader.GetGDCMSeriesFileNames(patient_path)

    print(slice_names)

    # slice_names=sorted(slice_names)
    reader.SetFileNames(slice_names)
    image = reader.Execute()

    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    # keys = image.GetMetaDataKeys()

    image_array = sitk.GetArrayFromImage(image)  # (Depth, Height, Width)
    return image_array


def nii_reader(patient_path):
    img_itk = sitk.ReadImage(patient_path)
    image_array = sitk.GetArrayFromImage(img_itk)
    return image_array


path_img = '/home/amax/Wendy/Kaggle/brain_tumor_radiogenomic_clf/registration_files/Train'
path_mask = '/home/amax/Wendy/nnUNet/ouput_kaggle/data_predict'
modality = 'T2w'
data_delete = ['00014', '00098', '00113', '00149', '00140', '00170', '00185', '00212', '00218', '00243', '00540', '00578', '00731', '00751', '00802', '00839']
print('data_delete', data_delete)
path_save = '/home/amax/Wendy/Kaggle/brain_tumor_radiogenomic_clf/roi_npy_train'
path_save_png = '/home/amax/Wendy/Kaggle/brain_tumor_radiogenomic_clf/roi_npy_png'

patient_name_ori = os.listdir(path_img)
print('patient_name_ori', patient_name_ori)
patient_name_img = [os.path.join(path_img, i, 'T2w', 'T2w_registered.nii.gz') for i in patient_name_ori]
print('patient_name_img', len(patient_name_img), patient_name_img)
patient_name_mask = os.listdir(path_mask)
patient_name_mask = [os.path.join(path_mask, i) for i in patient_name_mask if i.endswith('.nii.gz')]
print('patient_name_mask', patient_name_mask)

count = 0
for patient_path in patient_name_img:
    print('patient_path', patient_path)
    ID = patient_path.split('/')[-3]
    print('ID', ID)
    patient_path_mask = os.path.join(path_mask, ID + '.nii.gz')
    print('patient_path_mask', patient_path_mask)
    if os.path.isfile(patient_path_mask) & os.path.isfile(patient_path):
        if ID not in data_delete:
            # print('yes')
            count += 1
            image_array = nii_reader(patient_path)
            image_array = (image_array-np.min(image_array))/(np.max(image_array) - np.min(image_array))
            print('image_array', image_array.shape)
            image_array_mask = nii_reader(patient_path_mask)
            print('image_array_mask', image_array_mask.shape)
            # image_array_mask[image_array_mask > 2] = 0
            # image_array_mask[image_array_mask < 2] = 0
            image_array_mask[image_array_mask > 0] = 1
            print('image_array_mask', np.max(image_array_mask), np.min(image_array_mask))
            roi_img = image_array * image_array_mask
            print('roi_img', np.max(roi_img), np.min(roi_img))
            slice_sum = []
            slice_idx = []
            for slice in range(roi_img.shape[0]):
                slice_sum.append(np.sum(roi_img[slice, :, :]))
                slice_idx.append(slice)
            max_sum = np.max(slice_sum)
            idx_max = slice_sum.index(max_sum)
            slice_num = slice_idx[idx_max]
            roi_img_new = np.zeros((20, 240, 240))
            for idx in range(20):
                if (slice_num-10+idx < 155) & (slice_num-10+idx >= 0):
                    roi_img_new[idx, :, :] = roi_img[slice_num-10+idx, :, :]
                else:
                    roi_img_new[idx, :, :] = np.zeros(roi_img[0, :, :].shape)

            plt.imshow(np.concatenate((roi_img_new[10, :, :], image_array[slice_num, :, :]), axis=1), cmap='gray')
            # plt.show()
            # plt.imshow(image_array_mask[slice_num, :, :], cmap='gray')
            # plt.show()
            plt.savefig(os.path.join(path_save_png, ID))
            plt.close()
            np.save(os.path.join(path_save, ID), roi_img_new)

print('count', count)



