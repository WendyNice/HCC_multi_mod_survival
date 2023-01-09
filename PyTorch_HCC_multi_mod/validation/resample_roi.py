import os
import shutil
import nibabel as nib
import nibabel.processing as nibproc
import numpy as np
import concurrent.futures
import SimpleITK as sitk
import matplotlib.pyplot as plt


def slice_select(roi_img):
    # print('sum', np.sum(roi_img))
    sum_all = []
    idx_all = []
    for idx in range(roi_img.shape[2]):
        if np.sum(roi_img[:, :, idx]) > 0:
            # print('idx', idx)
            sum_all.append(np.sum(roi_img[:, :, idx]))
            idx_all.append(idx)
    print('sum_all', len(sum_all))
    if len(sum_all) == 0:
        return False, 0, 0
    else:
        max_idx = sum_all.index(max(sum_all))
        slice_num = idx_all[max_idx]
        print('slice_num', slice_num)
        return True, slice_num, idx_all


def resampling_reorientation(name, mod, img, output, out_shape):
    # resampling to 240x240x155 1mm3
    resampledIm = nibproc.conform(img, out_shape=out_shape, voxel_size=(1.0, 1.0, 1.0), order=1, cval=0.0)  #双线性插值重新采样
    # Modify metadata for reorientation
    n1_header = img.header
    # print(n1_header)
    n1_header.set_sform(np.diag([0, 0, 0, 0]), code='unknown')
    qform = np.array(
        [[1., 0., 0., -120.],
         [0., 1., 0., -129.],
         [0., 0., 1, -68.],
         [0., 0., 0., 1.]])

    n1_header.set_qform(qform, code='scanner')
    # print('--------------------------')
    # print(n1_header)
    img_data = img.get_data()
    print('img_data', img_data.shape, np.sum(img_data))

    # Reoriented to SRI24 t1 Atlas
    new_img = nib.nifti1.Nifti1Image(resampledIm.get_fdata(), None, header=n1_header)
    # print('new_img', new_img.get_data())
    show_img = new_img.get_data()
    # print('type', type(show_img))
    print('show_img', show_img.shape, np.sum(show_img))

    status, slice_num, idx_all = slice_select(show_img)
    if status:
        plt.imshow(show_img[:, :, slice_num], cmap='gray')
        plt.title('resample' + str(slice_num))
        plt.savefig(os.path.join('/data/Wendy/HCC/valid_set/png_roi', name + '_' + mod + '_resample.png'))
        plt.close()
        status, slice_num, idx_all = slice_select(img_data)
        plt.imshow(img_data[:, :, slice_num], cmap='gray')
        plt.title('ori' + str(slice_num))
        plt.savefig(os.path.join('/data/Wendy/HCC/valid_set/png_roi', name + '_' + mod + '_ori.png'))
        plt.close()
        nib.save(new_img, output)
        return True
    else:
        print('not satisfied')
        print(name)
        return False


def move_file(src_dir, tar_dir):
    if not os.path.exists(tar_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(tar_dir)
    shutil.copy(src_dir, tar_dir)


def processing(src_file, path_output, mod, out_shape):
    tar_file = os.path.join(path_output, src_file.split('/')[-3], mod, src_file.split('/')[-1])
    print('tar_file', tar_file)
    img = nib.load(src_file)
    array_data = img.get_data()
    array_img = nib.Nifti1Image(array_data, img.affine)
    satistied_status = resampling_reorientation(src_file.split('/')[-3], mod, array_img, tar_file, out_shape)
    if not satistied_status:
        return False
    else:
        return True


def resample_mask(path, patients, path_output, mod, out_shape):
    not_satisfied_pat = []
    for patient in patients:
        path_AP = os.path.join(path, patient, mod)
        src_file = os.path.join(path_AP, str(patient) + mod.replace('WI', '') + '.nii.gz')
        # print('src_file', src_file)
        status = processing(src_file, path_output, mod, out_shape)
        if status:
            pass
        else:
            not_satisfied_pat.append(src_file)
    return not_satisfied_pat


path = '/data/Wendy/HCC/valid_set/data_sort'
path_output = '/data/Wendy/HCC/valid_set/resample'

patient_ls = os.listdir(path)

out_shape = (512, 512, 150)
# img startwith data, mask startwith mask



not_satisfied_pat_AP = resample_mask(path, patient_ls, path_output, mod='AP', out_shape=out_shape)
not_satisfied_pat_PVP = resample_mask(path, patient_ls, path_output, mod='PVP', out_shape=out_shape)
not_satisfied_pat_T1WI = resample_mask(path, patient_ls, path_output, mod='T1WI', out_shape=out_shape)

print('not_satisfied_pat_AP', len(not_satisfied_pat_AP), not_satisfied_pat_AP)
print('not_satisfied_pat_PVP', len(not_satisfied_pat_PVP), not_satisfied_pat_PVP)
print('not_satisfied_pat_T1WI', len(not_satisfied_pat_T1WI), not_satisfied_pat_T1WI)
